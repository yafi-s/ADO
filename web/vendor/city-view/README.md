# City viewer dependencies

`three-view.js` bundles the selected Three.js 0.176.0 exports in `entry.mjs`, including GLTFLoader, OrbitControls and RoomEnvironment. Three.js is MIT licensed. The page loads this file locally and requires no package installation or external asset host.

To reproduce the checked-in bundle in a temporary directory, copy `entry.mjs` there and run:

```sh
npm install --no-save --no-audit --no-fund three@0.176.0 esbuild@0.25.5
npx --no-install esbuild entry.mjs --bundle --format=esm --minify --legal-comments=eof --outfile=three-view.js
```

Copy the resulting bundle and `node_modules/three/LICENSE` back into this directory.
