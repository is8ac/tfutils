package descend

import (
	"fmt"
	"testing"

	"github.com/is8ac/tfutils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func TestNewSeedSM(t *testing.T) {
	paramDefs := []ParamDef{
		ParamDef{Name: "foo", Init: tfutils.Zero(tf.Float, tf.MakeShape(1, 2))},
		ParamDef{Name: "bar", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
	}
	// make the first state machine
	s1 := op.NewScope()
	noise := MakeNoise(0.003)
	makeSM1, _, _, smParams1 := NewSeedSM(s1.SubScope("sm"), noise, paramDefs, 5)
	graph1, err := s1.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess1, err := tf.NewSession(graph1, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm1, err := makeSM1(sess1)
	if err != nil {
		t.Fatal(err)
	}

	// make the first state machine
	s2 := op.NewScope()
	makeSM2, _, _, smParams2 := NewSeedSM(s2.SubScope("sm"), noise, paramDefs, 5)
	graph2, err := s2.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess2, err := tf.NewSession(graph2, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm2, err := makeSM2(sess2)
	if err != nil {
		t.Fatal(err)
	}
	// now we can run them side by side to see if they act the same
	err = sm1.Step(3)
	if err != nil {
		t.Fatal(err)
	}
	err = sm2.Step(3)
	if err != nil {
		t.Fatal(err)
	}
	params1, err := sess1.Run(nil, smParams1, nil)
	if err != nil {
		t.Fatal(err)
	}
	params2, err := sess2.Run(nil, smParams2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if params1[0].Value().([][]float32)[0][0] != params2[0].Value().([][]float32)[0][0] {
		t.Fatal("params are different")
	}
	err = sm2.Step(7)
	if err != nil {
		t.Fatal(err)
	}
	err = sm1.Step(5)
	if err != nil {
		t.Fatal(err)
	}
	err = sm1.Rewind()
	if err != nil {
		t.Fatal(err)
	}
	err = sm1.Step(7)
	if err != nil {
		t.Fatal(err)
	}
	params1, err = sess1.Run(nil, smParams1, nil)
	if err != nil {
		t.Fatal(err)
	}
	params2, err = sess2.Run(nil, smParams2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if params1[0].Value().([][]float32)[0][0] != params2[0].Value().([][]float32)[0][0] {
		fmt.Println(sm1.Seeds, sm2.Seeds, sm1.Generation, sm2.Generation)
		fmt.Println(params1[0].Value().([][]float32)[0][0], params2[0].Value().([][]float32)[0][0])
		t.Fatal("params are different")
	}
}

func TestNewBestSeed(t *testing.T) {
	paramDefs := []ParamDef{
		ParamDef{Name: "foo", Init: tfutils.Zero(tf.Float, tf.MakeShape(1, 2))},
		ParamDef{Name: "bar", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
	}
	lossFunc := func(s *op.Scope, params []tf.Output) (loss tf.Output) {
		loss = op.Const(s, float32(0.1))
		return
	}
	s := op.NewScope()
	noise := MakeNoise(0.003)
	makeSM, newBestSeed, _, _ := NewSeedSM(s.SubScope("sm"), noise, paramDefs, 30)
	makeBestSeed := newBestSeed(lossFunc)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm, err := makeSM(sess)
	if err != nil {
		t.Fatal(err)
	}
	bestSeed, err := makeBestSeed(sess)
	if err != nil {
		t.Fatal(err)
	}
	seed, err := bestSeed()
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(seed)
	err = sm.Step(seed)
	if err != nil {
		t.Fatal(err)
	}
	seed, err = bestSeed()
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(seed)
}

func TestSeedSMtrain(t *testing.T) {
	s := op.NewScope()

	y := op.Const(s.SubScope("x"), []float32{1, 2, 3, 4})
	x := op.Const(s.SubScope("y"), []float32{0, -1, -2, -3})
	lossFunc := func(s *op.Scope, params []tf.Output) (loss tf.Output) {
		actual := op.Add(s, params[1], op.Mul(s, x, params[0]))
		return op.Sum(s, op.SquaredDifference(s, y, actual), op.Const(s.SubScope("reduction_indices"), []int32{0}))
	}
	paramDefs := []ParamDef{
		ParamDef{Name: "weight", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
		ParamDef{Name: "bias", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
	}
	noise := MakeNoise(0.003)                                                          // make the func to make noise
	makeSM, newBestSeed, _, params := NewSeedSM(s.SubScope("sm"), noise, paramDefs, 5) // make the state machine.
	makeBestSeed := newBestSeed(lossFunc)                                              // make the ops to get calculate the best seed.
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
	paramTensors, err := sess.Run(nil, params, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 500; i++ { // for 100 generations,
		seed, err := bestSeed()
		if err != nil {
			t.Fatal(err)
		}
		err = sm.Step(seed)
		if err != nil {
			t.Fatal(err)
		}
	}
	paramTensors, err = sess.Run(nil, params, nil)
	if err != nil {
		t.Fatal(err)
	}
	weight := paramTensors[0].Value().(float32)
	bias := paramTensors[1].Value().(float32)
	if weight < -1.1 || weight > -0.9 {
		fmt.Println("weight:", weight)
		t.Fatal("weight is not ~-1")
	}
	if bias > 1.1 || bias < 0.9 {
		fmt.Println("bias:", bias)
		t.Fatal("bias is not ~1")
	}
}

func TestNewWeightedSeedSM(t *testing.T) {
	paramDefs := []ParamDef{
		ParamDef{Name: "foo", Init: tfutils.Zero(tf.Float, tf.MakeShape(1, 2))},
		ParamDef{Name: "bar", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
	}
	// make the first state machine
	s1 := op.NewScope()
	noise := MakeNoise(0.003)
	makeSM1, _, _, smParams1 := NewWeightedSeedSM(s1.SubScope("sm"), noise, paramDefs, 2)
	graph1, err := s1.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess1, err := tf.NewSession(graph1, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm1, err := makeSM1(sess1)
	if err != nil {
		t.Fatal(err)
	}

	// make the first state machine
	s2 := op.NewScope()
	makeSM2, _, _, smParams2 := NewWeightedSeedSM(s2.SubScope("sm"), noise, paramDefs, 2)
	graph2, err := s2.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess2, err := tf.NewSession(graph2, nil)
	if err != nil {
		t.Fatal(err)
	}
	sm2, err := makeSM2(sess2)
	if err != nil {
		t.Fatal(err)
	}
	// now we can run them side by side to see if they act the same
	err = sm1.Step([]float32{0.2, 0.8})
	if err != nil {
		t.Fatal(err)
	}
	err = sm2.Step([]float32{0.2, 0.8})
	if err != nil {
		t.Fatal(err)
	}
	err = sm1.Step([]float32{0.6, 0.4})
	if err != nil {
		t.Fatal(err)
	}
	err = sm2.Step([]float32{0.6, 0.4})
	if err != nil {
		t.Fatal(err)
	}
	params1, err := sess1.Run(nil, smParams1, nil)
	if err != nil {
		t.Fatal(err)
	}
	params2, err := sess2.Run(nil, smParams2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if params1[0].Value().([][]float32)[0][0] != params2[0].Value().([][]float32)[0][0] {
		t.Fatal("params are different")
	}
	params1, err = sess1.Run(nil, smParams1, nil)
	if err != nil {
		t.Fatal(err)
	}
	params2, err = sess2.Run(nil, smParams2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if params1[0].Value().([][]float32)[0][0] != params2[0].Value().([][]float32)[0][0] {
		fmt.Println(sm1.SeedWeights, sm2.SeedWeights, sm1.Generation, sm2.Generation)
		fmt.Println(params1[0].Value().([][]float32)[0][0], params2[0].Value().([][]float32)[0][0])
		t.Fatal("params are different")
	}
}

func TestWeightedSeedSMtrain(t *testing.T) {
	s := op.NewScope()

	y := op.Const(s.SubScope("x"), []float32{1, 2, 3, 4})
	x := op.Const(s.SubScope("y"), []float32{0, -1, -2, -3})
	lossFunc := func(s *op.Scope, params []tf.Output) (loss tf.Output) {
		actual := op.Add(s, params[1], op.Mul(s, x, params[0]))
		return op.Sum(s, op.SquaredDifference(s, y, actual), op.Const(s.SubScope("reduction_indices"), []int32{0}))
	}
	paramDefs := []ParamDef{
		ParamDef{Name: "weight", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
		ParamDef{Name: "bias", Init: tfutils.Zero(tf.Float, tf.ScalarShape())},
	}
	noise := MakeNoise(0.003)                                                                      // make the func to make noise
	makeSM, newSeedWeights, _, params := NewWeightedSeedSM(s.SubScope("sm"), noise, paramDefs, 5)  // make the state machine.
	makeSeedWeights := newSeedWeights(lossFunc, op.Const(s.SubScope("seed_weight"), float32(100))) // make the ops to get calculate the seed weights.
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
	seedWeights, err := makeSeedWeights(sess) // Finalize the seed evaluator
	if err != nil {
		t.Fatal(err)
	}
	paramTensors, err := sess.Run(nil, params, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ { // for 1000 generations,
		weights, err := seedWeights()
		if err != nil {
			t.Fatal(err)
		}
		err = sm.Step(weights)
		if err != nil {
			t.Fatal(err)
		}
	}
	paramTensors, err = sess.Run(nil, params, nil)
	if err != nil {
		t.Fatal(err)
	}
	weight := paramTensors[0].Value().(float32)
	bias := paramTensors[1].Value().(float32)
	if weight < -1.1 || weight > -0.9 {
		fmt.Println("weight:", weight)
		t.Fatal("weight is not ~-1")
	}
	if bias > 1.1 || bias < 0.9 {
		fmt.Println("bias:", bias)
		t.Fatal("bias is not ~1")
	}
}
