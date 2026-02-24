declare module '*.wgsl' {
    const content: string;
    export default content;
}

declare module '*?raw' {
    const content: string;
    export default content;
}
