package main

import (
	"fmt"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/accuracy"
	"github.com/is8ac/tfutils/es/loss"
	"github.com/is8ac/tfutils/es/models"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	const seed int64 = 42
	err := mnist.Download()
	if err != nil {
		panic(err)
	}
	s := op.NewScope()

	// we create queue of batches.
	images, labels, init := mnist.NextBatch(s.SubScope("next_batch"),
		func(s *op.Scope, input tf.Output) tf.Output { // give it a func to modify the images. We want flat floats, so we must put two transformer funcs together.
			return mnist.InitCastImages(tf.Float)(s, // Cast to float32 and rescale from 0-255 to 0-1
				mnist.FlattenImages(s, input), // flatten from `[?, 28, 28]` to `[?, 784]`.
			)
		},
		mnist.InitOneHotLabels(tf.Float), // Let's also onehot encode the labels when training.
		300,  // Evaluate the model on 300 images each iteration.
		seed, // seed for repeatability
	)
	// Now we need to get test data to measure accurecy. We look at the whole test set, so no need for queues.
	initTestImages, testImages := tfutils.VarCache(s.SubScope("testImages"), // The test images are flattened, but are left as `uint8`s.
		mnist.InitCastImages(tf.Float)(s, mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s))),
		"test_images",
	)
	initTestLabels, testLabels := tfutils.VarCache(s.SubScope("testLabels"), // The test labels are however unaltered uint8.
		mnist.LabelsTest(s),
		"test_labels",
	)
	// bundle the init OPs to be passed to es.NewSession
	initOPs := []*tf.Operation{
		init,
		initTestLabels,
		initTestImages,
	}
	// Now we finaly get to construct the es session.
	esSess, err := es.NewSession(s.SubScope("main"),
		models.SingleLayerNN(tf.Float, 28*28, 10), // construct the graph def. It will take 784 inputs, and produce 10 outputs.
		loss.SoftmaxSqrDif,                        // We use squared difference on softmax to calculate loss.
		accuracy.Percent,                          // Accuracy is calculated as percent top 1.
		images,                                    // read a batch of images.
		labels,                                    // read a batch of labels.
		testImages,                                // read the test images,
		testLabels,                                // and the test labels.
		initOPs,                                   // slice of OPs to be called once to init the dataset and the var cache.
		3,                                         // evaluate 30 children each generation
		seed,                                      // seed for repeatability
		"tb_logs",                                 // dir in which to store tensorboard logs
		"run8",                                    // run name
	).Finalize()
	if err != nil {
		panic(err)
	}
	if err = esSess.WriteTBgraph("tb_logs/single_layer"); err != nil {
		panic(err)
	}
	for i := 0; i < 10000; i++ {
		bestIndex, err := esSess.BestIndex() // get the seed which, when used to perturb the params, gives the best loss.
		if err != nil {
			panic(err)
		}
		err = esSess.Perturb(bestIndex) // Then perturb the params with it.
		if err != nil {
			panic(err)
		}
		if i%30 == 0 { // every 30th iteration, test with the test set, and log both accuracy and the graph log OPs.
			acc, err := esSess.Accuracy(true, true)
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc*100, "%") // Log the accuracy.
		}
	}
	// You should get to ~50% accuracy after 100 iterations.
}
