package main

import (
	"fmt"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/accuracy"
	"github.com/is8ac/tfutils/es/loss"
	"github.com/is8ac/tfutils/es/models"
	"github.com/is8ac/tfutils/es/models/rnn"
	"github.com/is8ac/tfutils/mnist"
	"github.com/is8ac/tfutils/text"
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
	const seqLen int64 = 20
	const batchSize int64 = 25
	const testNumSeq int64 = 31
	const stateSize int64 = 100
	chars, nextChars, init := text.NextSeqBatch(s.SubScope("training"), "snark.txt", seqLen, batchSize, 500, 42)
	readTestInput, readTestTargets := text.Offset(s.SubScope("offset_test"), text.ReadText(s.SubScope("read_test"), "jabberwock.txt"))
	initTestInputs, testInputs := tfutils.VarCache(s.SubScope("test_inputs"), text.OneHot(s, text.Split(s.SubScope("ti"), readTestInput, seqLen, testNumSeq)), "test_images")
	initTestTargets, testTargets := tfutils.VarCache(s.SubScope("test_targets"), text.Split(s.SubScope("tt"), readTestTargets, seqLen, testNumSeq), "test_labels")
	// bundle the init OPs to be passed to es.NewSession
	initOPs := []*tf.Operation{
		init,
		initTestInputs,
		initTestTargets,
	}
	cellDef := rnn.LSTM(tf.Float, 256, stateSize).AddPost(models.SingleLayerNN(tf.Float, stateSize, 256))
	// Now we finaly get to construct the es session.
	esSess, err := es.NewSession(s.SubScope("main"),
		rnn.Unroll(seqLen, cellDef),
		loss.SoftmaxSqrDif, // We use squared difference on softmax to calculate loss.
		accuracy.Percent,   // Accuracy is calculated as percent top 1.
		chars,              // read a batch of chars.
		nextChars,          // read a batch of next chars.
		testInputs,
		testTargets, // and the test labels.
		initOPs,     // slice of OPs to be called once to init the dataset and the var cache.
		30,          // evaluate 30 children each generation
		seed,        // seed for repeatability
		"tb_logs/rnn",
		"",
	).Finalize()
	if err != nil {
		panic(err)
	}
	//err = tb.WriteGraphSummary(esSess.Graph, "tb_logs")
	//if err != nil {
	//	panic(err)
	//}
	for i := 0; i < 10000; i++ { // for 100 generations,
		bestIndex, err := esSess.BestIndex() // get the seed which, when used to perturb the params, gives the best loss.
		if err != nil {
			panic(err)
		}
		err = esSess.Perturb(bestIndex) // Then perturb the params with it.
		if err != nil {
			panic(err)
		}
		if i%10 == 0 {
			acc, err := esSess.Accuracy(false, false)
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc*100, "%") // Log the accuracy.
		}
	}
}
