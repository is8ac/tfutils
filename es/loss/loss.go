// Package loss provides functions to calculate how much two tensors fail to match.
package loss

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// SoftmaxSqrDif uses softmax to calculate loss
func SoftmaxSqrDif(s *op.Scope, actual, target tf.Output) (loss tf.Output) {
	if actual.Shape().NumDimensions() == 3 {
		dims, err := actual.Shape().ToSlice()
		if err != nil {
			panic(err)
		}
		shape := op.Const(s.SubScope("reshape_dims"), []int64{-1, dims[len(dims)-1]})
		actual = op.Reshape(s.SubScope("actual"), actual, shape)
		target = op.Reshape(s.SubScope("actual"), target, shape)
	}
	softmax := op.Softmax(s, actual)
	sqrDiffs := op.SquaredDifference(s, softmax, target)
	sums := op.Sum(s, sqrDiffs, op.Const(s, int32(-1)))
	loss = op.Mean(s, sums, op.Const(s.SubScope("mean_reduce_dims"), []int32{0}))
	return
}

func softmaxCrossEntropyWithLogits(s *op.Scope, actual, labels tf.Output) (loss tf.Output) {
	_, loss = op.SoftmaxCrossEntropyWithLogits(s, actual, labels)
	return
}
