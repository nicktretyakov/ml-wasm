@echo off
echo Building WebAssembly module...
wasm-pack build --target web

echo Build complete! You can now serve the project with a web server.
echo For example: python -m http.server
