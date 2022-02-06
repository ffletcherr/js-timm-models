const iterations = 20;
const warmup = 20;
const backend = "wasm" //"webgl" or "wasm"
const input_shape = [1, 3, 224, 224]
const model_name = "efficientnet.onnx"

async function main() {
    const net = await ort.InferenceSession.create(model_name, { executionProviders: [backend] });
    const tensor = new ort.Tensor('float32', new Float32Array(input_shape.reduce((a, b)=> a*b, 1)), input_shape);

    // Warming Up !
    console.log("Start Warming Up for %s iterations", warmup);
    let t1 = new Date();
    for (let i = 0; i < warmup; i++) {
        const output = await net.run({ ["input"]: tensor });
    }
    let t2 = new Date();
    console.log("average warmup time:", (t2 - t1) / warmup, "ms")

    // Inference !
    console.log("Start Inference for %s iterations", iterations);
    for (let i = 0; i < iterations; i++) {

        const output = await net.run({ ["input"]: tensor });
    }
    let t3 = new Date();
    console.log("average inference time:", (t3 - t2) / iterations, "ms")
}


main()