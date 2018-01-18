# Evolutionary Strategy optimizer

Uses ES to optimize the parameters of arbitrary compute graphs.

## Examples
Let us train a single layer fully interconnected neural net to classify the MNIST data set.
This is equivalent to the [MNIST for ML Beginners](https://www.tensorflow.org/get_started/mnist/beginners) tutorial in the TensorFlow docs.

Import:
```
import (
	"fmt"
	"time"

	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)
```
Create the scope:
```
s := op.NewScope()
```
Create operations to read the training data:
```
images, labels, init := mnist.NextBatch(s.SubScope("next_batch"),
  func(s *op.Scope, input tf.Output) tf.Output {
    return mnist.InitCastImages(dType)(s, mnist.FlattenImages(s, input))
  },
  mnist.InitOneHotLabels(dType),
  300,
  42,
)
```
`mnist.NextBatch` takes:
- a scope
- a function to transform images
- a function to transform labels
- the batch size
- a seed for deterministic pseudo-randomness

To transform the images, we pass a function which flattens the images `[?, 28, 28]` images to `[?, 784]`, and then casts them to

Each time the `images` output is pulled on, it will return a tensor of shape `[300, 748]`.
Each time the `labels`

```
initTestLabels, testLabels := varCache(s.SubScope("testLabels"), mnist.LabelsTest(s), "test_labels")
initTestImages, testImages := varCache(s.SubScope("testImages"), mnist.InitCastImages(dType)(s, mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s))), "test_images")
initOPs := []*tf.Operation{
  init,
  initTestLabels,
  initTestImages,
}
```


If we want to optimize the parameters of a simple single layer fully interconnected neural net:

First, describe the variables that can be perturbed.
```
vars = []es.VarDef{
  es.VarDef{Name: "weights", DataType: tf.Float, Shape: tf.MakeShape(784, 10), PerturbFunc: es.MakeSlicePerturb(0.003, tf.Float)},
  es.VarDef{Name: "biases", DataType: tf.Float, Shape: tf.MakeShape(10), PerturbFunc: es.MakeSlicePerturb(0.003, tf.Float)},
}
```
- Name: Must be unique within each model. Be descriptive so as to help debugging.
- DataType: A tensorflow.DataType. In this case, a 32 bit float.
- Shape: A tensorflow.Shape.
- PerturbFunc: A function that takes a scope and a tf.Output, and returns a tf.Output. In this example, we call the built in `es.MakeSlicePerturb()` which returns a suitable PerturbFunc. Note that the DataType set in the varDef must match that of the PerturbFunc. Here we tell `es.MakeSlicePerturb()` to make us a PerturbFunc which takes `tf.Float`s.


Next, we write a function to make the actual model.
This function must take a scope, a slice of tf.Outputs, and an output.
```
modelFunc := func(s *op.Scope, vars []tf.Output, inputs tf.Output) (output tf.Output) {
  output = op.Add(s, vars[1], op.MatMul(s, inputs, vars[0]))
  return
}
```
In this example, `inputs` is of shape `[?, 784]`.
In this simple single layer NN, we matmul the inputs, which in this case are of shape `[?, 784]` with the weights variable, which we defined as `[784, 10]`, and get an output of shape `[?, 10]`.
This output we op.Add with the biases to get the final output which return.

Next we
