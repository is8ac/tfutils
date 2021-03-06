package main

import (
	"fmt"
	"strconv"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/descend"
	"github.com/is8ac/tfutils/descend/models"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	const numSeeds = 50
	const globalSeed = 42
	const batchSize = 400
	const noiseStdev float32 = 0.003
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
		batchSize,                        // Evaluate the model on 400 images each iteration.
		globalSeed,                       // seed for repeatability
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

	noise := descend.MakeNoise(noiseStdev)                                                                       // make the func to make noise
	paramDefs, lossFunc, makeFinalizeAccuracy := models.MakeSingleLayerNN(images, labels)                        // create the funcs to evaluate loss
	makeSM, newBestSeed, generation, params := descend.NewSeedSM(s.SubScope("sm"), noise, paramDefs, numSeeds)   // make the state machine.
	makeBestSeed := newBestSeed(lossFunc)                                                                        // make the ops to get calculate the best seed.
	finalizeAccuracy, accuracyOP := makeFinalizeAccuracy(s.SubScope("accuracy"), params, testImages, testLabels) // give the accuracy func params and some test data.

	loggingScope := s.SubScope("logging")
	writer := op.SummaryWriter(loggingScope, op.SummaryWriterSharedName("tb_logs"))
	createSummaryWriter := op.CreateSummaryFileWriter(loggingScope,
		writer,
		op.Const(loggingScope.SubScope("log_dir"), "tb_logs/single_layer/es/"+strconv.Itoa(numSeeds)+"seeds_batchsize"+strconv.Itoa(batchSize)+"_noise"+strconv.FormatFloat(float64(noiseStdev), 'f', 4, 64)),
		op.Const(loggingScope.SubScope("max_queue"), int32(10)),
		op.Const(loggingScope.SubScope("flush_millis"), int32(100)),
		op.Const(loggingScope.SubScope("filename_suffix"), ".tblog"),
	)
	logAcc := op.WriteScalarSummary(loggingScope, writer, generation, op.Const(s.SubScope("acc_tag"), "accuracy"), accuracyOP)
	logWeightsHist := op.WriteHistogramSummary(loggingScope, writer, generation, op.Const(loggingScope.SubScope("weights_hist_tag"), "weights"), params[0])
	closeSummaryWriter := op.CloseSummaryWriter(loggingScope, writer)

	graph, err := s.Finalize()
	if err != nil {
		panic(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	sm, err := makeSM(sess) // Finalize the state machine
	if err != nil {
		panic(err)
	}
	bestSeed, err := makeBestSeed(sess) // Finalize the seed evaluator
	if err != nil {
		panic(err)
	}
	_, err = sess.Run(nil, nil, initOPs) // initialize the data vars
	if err != nil {
		panic(err)
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{createSummaryWriter})
	if err != nil {
		panic(err)
	}
	accuracy := finalizeAccuracy(sess) // finally make the accuracy func.
	for i := 0; i < 2000; i++ {        // for 1000 generations,
		seed, err := bestSeed()
		if err != nil {
			panic(err)
		}
		//fmt.Println(seed)
		err = sm.Step(seed)
		if err != nil {
			panic(err)
		}
		if i%10 == 0 {
			acc, err := accuracy()
			if err != nil {
				panic(err)
			}
			_, err = sess.Run(nil, nil, []*tf.Operation{logAcc, logWeightsHist})
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc*100, "%")
		}
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{closeSummaryWriter})
	if err != nil {
		panic(err)
	}
	// You should get to ~50% accuracy after 100 iterations.
}
