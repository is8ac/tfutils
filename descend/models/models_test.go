package models

import (
	"testing"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/descend"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func TestSingleLayer(t *testing.T) {
	s := op.NewScope()

	// we create queue of batches.
	images, labels, init := mnist.NextBatch(s.SubScope("next_batch"),
		func(s *op.Scope, input tf.Output) tf.Output { // give it a func to modify the images. We want flat floats, so we must put two transformer funcs together.
			return mnist.InitCastImages(tf.Float)(s, // Cast to float32 and rescale from 0-255 to 0-1
				mnist.FlattenImages(s, input), // flatten from `[?, 28, 28]` to `[?, 784]`.
			)
		},
		mnist.InitOneHotLabels(tf.Float), // Let's also onehot encode the labels when training.
		400, // Evaluate the model on 300 images each iteration.
		42,  // seed for repeatability
	)
	// Now we need to get test data to measure accuracy. We look at the whole test set, so no need for queues.
	initTestImages, testImages := tfutils.VarCache(s.SubScope("testImages"), // The test images are flattened and cast to float32
		mnist.InitCastImages(tf.Float)(s, mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s))),
		"test_images",
	)
	initTestLabels, testLabels := tfutils.VarCache(s.SubScope("testLabels"), // The test labels are however unaltered uint8.
		op.Cast(s.SubScope("to_int32"), mnist.LabelsTest(s), tf.Int32),
		"test_labels",
	)
	// bundle the init OPs
	initOPs := []*tf.Operation{
		init,
		initTestLabels,
		initTestImages,
	}

	noise := descend.MakeNoise(0.003)                                                                     // make the func to make noise
	paramDefs, lossFunc, makeFinalizeAccuracy := MakeSingleLayerNN(images, labels)                        // create the funcs to evaluate loss
	makeSM, generation, params := descend.NewSeedSM(s.SubScope("sm"), noise, paramDefs)                   // make the state machine.
	makeBestSeed := descend.NewBestSeed(s.SubScope("best_seed"), params, lossFunc, noise, 50, generation) // make the ops to get calculate the best seed.
	finalizeAccuracy := makeFinalizeAccuracy(s.SubScope("accuracy"), params, testImages, testLabels)      // give the accuracy func params and some test data.
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm, err := makeSM(sess) // Finalize the state machine
	if err != nil {
		t.Fatal(err)
	}
	bestSeed, err := makeBestSeed(sess) // Finalize the seed evaluator
	if err != nil {
		t.Fatal(err)
	}
	_, err = sess.Run(nil, nil, initOPs) // initialize the data vars
	if err != nil {
		t.Fatal(err)
	}
	accuracy := finalizeAccuracy(sess) // finally. make the accuracy func.
	for i := 0; i < 10; i++ {
		seed, err := bestSeed()
		if err != nil {
			t.Fatal(err)
		}
		//fmt.Println(seed)
		err = sm.Step(seed)
		if err != nil {
			t.Fatal(err)
		}
		if i%1 == 0 {
			_, err := accuracy()
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	return
}
