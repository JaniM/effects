use std::fs::File;
use std::io::Read;

use lasso::Rodeo;
use logos::{Logos, Span};

use petgraph::stable_graph::StableGraph;

type StringKey = lasso::Spur;

#[derive(Logos, Debug, PartialEq)]
enum Token {
    #[regex("[a-zA-Z][a-zA-Z_+*/-]*")]
    Symbol,

    #[token("(")]
    OpenBrace,

    #[token(")")]
    CloseBrace,

    #[regex("\"([^\"\\\\]|\\\\.)*\"")]
    String,

    #[regex("-?[0-9]+", |lex| lex.slice().parse())]
    Int(i64),

    #[regex("-?[0-9]+\\.[0-9]*", |lex| lex.slice().parse())]
    Float(f64),

    #[regex(r"[ \t\n\f]+", logos::skip)]
    #[error]
    Error,
}

#[derive(Debug, Clone)]
enum Literal {
    String(StringKey),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
enum Ast {
    Symbol(StringKey),
    Literal(Literal),
    List(Vec<Ast>)
}

#[derive(Debug, Clone)]
enum ParseError {
    UnexpectedClose {
        span: Span,
    },
    LexError {
        span: Span,
        slice: String
    },
    EOF
}

struct Parser<'a, 'b> {
    lexer: logos::Lexer<'a, Token>,
    interner: &'b mut Rodeo
}

impl<'a, 'b> Parser<'a, 'b> {
    fn new(source: &'a str, interner: &'b mut Rodeo) -> Self {
        let lexer = Token::lexer(source);
        Parser { lexer, interner }
    }

    fn parse(&mut self) -> Result<Ast, ParseError> {
        let token = self.lexer.next().ok_or(ParseError::EOF)?;
        self.parse_token(token)
    }

    fn parse_token(&mut self, token: Token) -> Result<Ast, ParseError> {
        let slice = self.lexer.slice();
        let ast = match token {
            Token::Symbol => Ast::Symbol(self.interner.get_or_intern(slice)),
            Token::String => Ast::Literal(Literal::String(self.interner.get_or_intern(&slice[1..slice.len()-1]))),
            Token::Float(v) => Ast::Literal(Literal::Float(v)),
            Token::Int(v) => Ast::Literal(Literal::Int(v)),
            Token::CloseBrace => return Err(ParseError::UnexpectedClose { span: self.lexer.span() }),
            Token::OpenBrace => self.parse_list()?,
            Token::Error => return Err(ParseError::LexError {
                span: self.lexer.span(),
                slice: slice.chars().take(20).collect()
            })
        };
        Ok(ast)
    }

    fn parse_list(&mut self) -> Result<Ast, ParseError> {
        let mut items = Vec::new();
        loop {
            let token = self.lexer.next().ok_or(ParseError::EOF)?;
            if token == Token::CloseBrace {
                break;
            }
            items.push(self.parse_token(token)?);
        }
        Ok(Ast::List(items))
    }
}

type NodeIndex = petgraph::graph::NodeIndex<u32>;
type EdgeIndex = petgraph::graph::EdgeIndex<u32>;
type NodeGraph = StableGraph<Node, ()>;

#[derive(Debug, Copy, Clone, PartialEq)]
enum Constant {
    String(StringKey),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
enum TypeKind {
    Unit,
    String
}

#[derive(Debug, Clone, PartialEq)]
struct TypeInfo {
    kind: TypeKind
}

#[derive(Debug, Clone, PartialEq)]
enum Operator {
    Symbol(StringKey),
    Constant(Constant),
    Call,
    Type(TypeInfo),
    FnDef { out: TypeInfo },
    ArgList,
    List
}

#[derive(Debug, Clone)]
struct Node {
    operator: Operator,
    parameters: Vec<NodeIndex>
}

impl Node {
    fn new(op: Operator, parameters: Vec<NodeIndex>) -> Self{
        Node {
            operator: op,
            parameters
        }
    }

    fn op0(op: Operator) -> Self {
        Node::new(op, Vec::new())
    }
}

fn ast_to_graph(graph: &mut NodeGraph, ast: &Ast) -> NodeIndex {
    match ast {
        Ast::Symbol(name) => {
            graph.add_node(Node::op0(Operator::Symbol(*name)))
        },
        Ast::Literal(v) => {
            graph.add_node(Node::op0(Operator::Constant(match v {
                Literal::String(v) => Constant::String(*v),
                Literal::Float(v) => Constant::Float(*v),
                Literal::Int(v) => Constant::Int(*v)
            })))
        },
        Ast::List(items) => {
            let mut nodes = Vec::new();
            for item in items {
                nodes.push(ast_to_graph(graph, item));
            }
            let node = graph.add_node(Node::new(Operator::List, nodes.clone()));
            for item in nodes {
                graph.add_edge(node, item, ());
            }
            node
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum TransformIndex {
    Old(NodeIndex),
    New(u32)
}

#[derive(Clone, Debug, PartialEq)]
enum Transform {
    Replace(NodeIndex, Operator),
    Connect(TransformIndex, TransformIndex),
    Disconnect(TransformIndex, TransformIndex),
    Create(Operator)
}

struct GraphTransformer<'a> {
    transforms: Vec<Transform>,
    follow: Vec<TransformIndex>,
    new_nodes: Vec<NodeIndex>,
    interner: &'a mut Rodeo,
    current_node: Option<NodeIndex>,
    new_count: u32
}

struct VisitContext<'a, 'b> {
    transformer: &'a mut GraphTransformer<'b>,
    graph: &'a NodeGraph,
    node: &'a Node
}

impl<'a> GraphTransformer<'a> {
    fn new(interner: &'a mut Rodeo) -> Self {
        GraphTransformer {
            transforms: Default::default(),
            follow: Default::default(),
            new_nodes: Default::default(),
            current_node: None,
            interner,
            new_count: 0
        }
    }

    fn follow(&mut self, idx: TransformIndex) {
        self.follow.push(idx);
    }

    fn follow_all(&mut self, indices: &[NodeIndex]) {
        self.follow.extend(indices.into_iter().map(|x| TransformIndex::Old(*x)));
    }

    fn resolve_string(&self, key: StringKey) -> &str {
        self.interner.resolve(&key)
    }

    fn replace_self(&mut self, op: Operator) {
        self.transforms.push(Transform::Replace(self.current_node.unwrap(), op));
    }

    fn replace(&mut self, idx: NodeIndex, op: Operator) {
        self.transforms.push(Transform::Replace(idx, op));
    }

    fn disconnect(&mut self, child: NodeIndex) {
        self.transforms.push(Transform::Disconnect(
            TransformIndex::Old(self.current_node.unwrap()),
            TransformIndex::Old(child)
        ));
    }

    fn resolve_index(&self, idx: TransformIndex) -> NodeIndex {
        Self::resolve_index_static(&self.new_nodes, idx)
    }

    fn resolve_index_static(new_nodes: &[NodeIndex], idx: TransformIndex) -> NodeIndex {
        match idx {
            TransformIndex::Old(idx) => idx,
            TransformIndex::New(idx) => new_nodes[idx as usize]
        }
    }
}

impl<'a, 'b> std::ops::Deref for VisitContext<'a, 'b> {
    type Target = GraphTransformer<'b>;

    fn deref(&self) -> &Self::Target {
        &self.transformer
    }
}

impl<'a, 'b> std::ops::DerefMut for VisitContext<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.transformer
    }
}

trait GraphVisitor {
    fn walk(&self, graph: &mut NodeGraph, interner: &mut Rodeo, root: NodeIndex) -> bool {
        let mut stack = vec![root];
        let mut transformer = GraphTransformer::new(interner);
        let mut transforms = Vec::new();
        while let Some(node_idx) = stack.pop() {
            let node = &graph[node_idx];
            transformer.current_node = Some(node_idx);
            self.visit(node, &mut transformer, graph);
            transforms.extend(transformer.transforms.drain(..));
            for transform in transforms.drain(..) {
                match transform {
                    Transform::Replace(idx, op) => {
                        graph[idx].operator = op;
                    },
                    Transform::Create(op) => {
                        transformer.new_nodes.push(graph.add_node(Node::op0(op)));
                    },
                    Transform::Connect(tidx1, tidx2) => {
                        let idx1 = transformer.resolve_index(tidx1);
                        let idx2 = transformer.resolve_index(tidx2);
                        graph.add_edge(idx1, idx2, ());
                        graph[idx1].parameters.push(idx2);
                    },
                    Transform::Disconnect(tidx1, tidx2) => {
                        let idx1 = transformer.resolve_index(tidx1);
                        let idx2 = transformer.resolve_index(tidx2);
                        if let Some(edge) = graph.find_edge(idx1, idx2) {
                            graph.remove_edge(edge);
                            graph[idx1].parameters.retain(|x| *x != idx2);
                        }
                    }
                }
            }
            let new_nodes = &transformer.new_nodes;
            stack.extend(transformer.follow.drain(..).map(|x|
                GraphTransformer::resolve_index_static(new_nodes, x)
            ));
            transformer.new_nodes.clear();
            transformer.new_count = 0;
        }
        true
    }

    fn visit(&self, node: &Node, transformer: &mut GraphTransformer, graph: &NodeGraph) {
        let ctx = VisitContext {
            transformer,
            graph,
            node
        };

        match &node.operator {
            Operator::Constant(_) => {},
            Operator::Symbol(_) => {},
            Operator::List => self.visit_list(ctx),
            Operator::Call => {},
            Operator::FnDef { .. } => self.visit_fndef(ctx),
            Operator::ArgList => {},
            Operator::Type(_) => self.visit_type(ctx),
        }
    }

    fn visit_list(&self, ctx: VisitContext) {
        ctx.transformer.follow_all(&ctx.node.parameters);
    }

    fn visit_fndef(&self, ctx: VisitContext) {
        ctx.transformer.follow_all(&ctx.node.parameters);
    }

    fn visit_type(&self, _ctx: VisitContext) { }
}

struct TransformPlainFnDefs;

impl GraphVisitor for TransformPlainFnDefs {
    fn visit_list(&self, mut ctx: VisitContext) {
        // (fn name args ret-type body) -> FnDecl(name ArgList(args) Type(ret-type) body)
        let parameters = &ctx.node.parameters;
        if parameters.len() == 0 {
            return;
        }

        let head = &ctx.graph[parameters[0]];
        let is_fn = match head.operator {
            Operator::Symbol(sym) => ctx.resolve_string(sym) == "fn",
            _ => false
        };
        if is_fn {
            ctx.replace_self(Operator::FnDef { out: TypeInfo { kind: TypeKind::Unit } });
            ctx.disconnect(parameters[0]);
            ctx.replace(parameters[2], Operator::ArgList);
            ctx.replace(parameters[3], Operator::Type(TypeInfo { kind: TypeKind::Unit }));
            ctx.follow(TransformIndex::Old(parameters[3]));
            ctx.follow(TransformIndex::Old(parameters[4]));
        } else {
            ctx.follow_all(parameters);
        }
    }

    fn visit_type(&self, mut ctx: VisitContext) {
        let parameters = &ctx.node.parameters;
        if parameters.len() == 0 {
            return;
        }

        let head = &ctx.graph[parameters[0]];
        let is_string = match head.operator {
            Operator::Symbol(sym) => ctx.resolve_string(sym) == "string",
            _ => false
        };

        if is_string {
            ctx.replace_self(Operator::Type(TypeInfo { kind: TypeKind::String }));
            ctx.disconnect(parameters[0]);
        }
    }
}

struct SolidifyFnTypes;

impl GraphVisitor for SolidifyFnTypes {
    fn visit_fndef(&self, ctx: VisitContext) {
        let t = &ctx.graph[ctx.node.parameters[2]];
        if let Operator::Type(info) = &t.operator {
            ctx.transformer.replace_self(Operator::FnDef { out: info.clone() });
            ctx.transformer.disconnect(ctx.node.parameters[2]);
        }
        ctx.transformer.follow(TransformIndex::Old(ctx.node.parameters[3]));
    }
}


fn print_graph(graph: &NodeGraph, interner: &Rodeo, idx: NodeIndex, depth: usize) {
    let node = &graph[idx];
    match &node.operator {
        Operator::Constant(c) => match c {
            Constant::Int(v) =>
                println!("{}Int {:?}", " ".repeat(depth), v),
            Constant::Float(v) =>
                println!("{}Float {:?}", " ".repeat(depth), v),
            Constant::String(key) =>
                println!("{}String {:?}", " ".repeat(depth), interner.resolve(key)),
        },
        Operator::Symbol(key) =>
            println!("{}Symbol {:?}", " ".repeat(depth), interner.resolve(key)),
        _ => println!("{}{:?}", " ".repeat(depth), node.operator)
    }
    for idx in &node.parameters {
        print_graph(graph, interner, *idx, depth + 2);
    }
}

fn main() {
    let code = {
        let mut code = String::new();
        File::open("foo.eff")
            .expect("File needed")
            .read_to_string(&mut code)
            .expect("Read failed");
        code
    };

    let mut interner = Rodeo::default();
    let ast = match Parser::new(&code, &mut interner).parse() {
        Ok(v) => v,
        Err(e) => {
            println!("{:?}", e);
            return;
        }
    };
    let mut graph = NodeGraph::default();
    let root = ast_to_graph(&mut graph, &ast);
    println!("{:?}", ast);
    print_graph(&graph, &interner, root, 0);
    TransformPlainFnDefs.walk(&mut graph, &mut interner, root);
    print_graph(&graph, &interner, root, 0);
    SolidifyFnTypes.walk(&mut graph, &mut interner, root);
    print_graph(&graph, &interner, root, 0);
}
