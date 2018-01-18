package perturb

import (
	"os"
	"testing"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func testPerturbFunc(pt es.PerturbFunc, zero func(*op.Scope) tf.Output, dims []int32) (err error) {
	s := op.NewScope()
	input := op.Fill(s, op.Const(s.SubScope("fill_dims"), dims), zero(s.SubScope("zero")))
	numOutputs := op.Const(s.SubScope("num_outputs"), int32(5))
	generation := op.Const(s.SubScope("gen"), int64(0))
	childPerturb := pt(s.SubScope("pt"), input, numOutputs, generation, 42)
	perturb0 := childPerturb(s.SubScope("child0"), op.Const(s.SubScope("index0"), int32(0)))
	perturb1 := childPerturb(s.SubScope("child1"), op.Const(s.SubScope("index1"), int32(1)))
	perturb2 := childPerturb(s.SubScope("child2"), op.Const(s.SubScope("index2"), int32(2)))
	graph, err := s.Finalize()
	if err != nil {
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	_, err = sess.Run(nil, []tf.Output{perturb0, perturb1, perturb2}, nil)
	if err != nil {
		return
	}
	return
}

func TestMakeSlice(t *testing.T) {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	shapes := [][]int32{[]int32{}, []int32{1}, []int32{5}, []int32{1, 5}, []int32{3, 5, 7}}
	dTypes := []tf.DataType{tf.Half, tf.Float, tf.Double}
	for _, shape := range shapes {
		for _, dType := range dTypes {
			if err := testPerturbFunc(MakeSlice(0.003), tfutils.Zero(dType), shape); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestMakeSimple(t *testing.T) {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	shapes := [][]int32{[]int32{}, []int32{1}, []int32{5}, []int32{1, 5}, []int32{3, 5, 7}}
	dTypes := []tf.DataType{tf.Half, tf.Float, tf.Double}
	for _, shape := range shapes {
		for _, dType := range dTypes {
			if err := testPerturbFunc(MakeSimple(0.003), tfutils.Zero(dType), shape); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestMakeQsimple(t *testing.T) {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	shapes := [][]int32{[]int32{}, []int32{1}, []int32{5}, []int32{1, 5}, []int32{3, 5, 7}}
	for _, shape := range shapes {
		if err := testPerturbFunc(MakeQsimple(0.003, 5), tfutils.Zero(tf.Quint8), shape); err != nil {
			t.Fatal(err)
		}
	}
}
