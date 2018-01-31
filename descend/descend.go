// Package descend provides tools to use gradient descent to optimize the parameters of continuous functions implemented as tensorflow graphs.
// It does not (yet) implement backpropagation, but rather uses ES and other black box optimization algorithms.
package descend

import (
	"strconv"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// LossFunc takes a slice of params and returns the loss for the slice
type LossFunc func(s *op.Scope, params []tf.Output) (loss tf.Output)

// NoiseFunc makes some deterministic noise.
type NoiseFunc func(s *op.Scope, shape tf.Output, seed, gen tf.Output) tf.Output

// ParamDef defines a shape and name
type ParamDef struct {
	Name string                    // Name must be unique
	Init func(*op.Scope) tf.Output // Output of the initial state
}

// ModelDef defines a model.
// Params describes the tensors of params which are to be optimised.
// Loss is a function which, when called, will create a sub graph which takes the params, and returns a loss value.
// Log takes params and generation, and returns a slice of operations to be pulled on.
type ModelDef struct {
	Params []ParamDef // The list of shapes of params that the child wants.
	Loss   LossFunc   // Loss takes a slice of params, and returns a number which will be minimized.
}

// MakeNoise creates a noise function
func MakeNoise(stdevVal float32) NoiseFunc {
	return func(s *op.Scope, shape tf.Output, seed, gen tf.Output) tf.Output {
		stdev := op.Const(s, stdevVal)
		noiseSeed := op.Pack(s, []tf.Output{gen, seed})
		oneNoise := op.StatelessRandomNormal(s, shape, noiseSeed, op.StatelessRandomNormalDtype(tf.Float))
		return op.Mul(s, oneNoise, stdev)
	}
}

// SeedSM allows one to move through the parameter space using seeds.
type SeedSM struct {
	Generation       int64
	Seeds            []int64
	perturb          []*tf.Operation
	deperturb        []*tf.Operation
	sess             *tf.Session
	seedPH           tf.Output
	genPH            tf.Output
	updateGeneration *tf.Operation
}

// Step moves the parameters through parameter space by one seed
func (sm *SeedSM) Step(seed int64) (err error) {
	sm.Generation++
	sm.Seeds = append(sm.Seeds, seed)
	seedTensor, err := tf.NewTensor(seed)
	if err != nil {
		panic(err)
	}
	genTensor, err := tf.NewTensor(sm.Generation)
	if err != nil {
		panic(err)
	}
	_, err = sm.sess.Run(map[tf.Output]*tf.Tensor{sm.seedPH: seedTensor, sm.genPH: genTensor}, nil, append(sm.perturb, sm.updateGeneration))
	if err != nil {
		return
	}
	return
}

// Rewind steps back by one step,
func (sm *SeedSM) Rewind() (err error) {
	seedTensor, err := tf.NewTensor(sm.Seeds[len(sm.Seeds)-1])
	if err != nil {
		panic(err)
	}

	genTensor, err := tf.NewTensor(sm.Generation)
	if err != nil {
		panic(err)
	}
	_, err = sm.sess.Run(map[tf.Output]*tf.Tensor{sm.seedPH: seedTensor, sm.genPH: genTensor}, nil, append(sm.deperturb, sm.updateGeneration))
	if err != nil {
		return
	}
	sm.Generation += -1
	sm.Seeds = sm.Seeds[:len(sm.Seeds)-1]
	return
}

// NewSeedSM creates TF OPs for a state machine to move through parameter space according to the seed which is give and the generation.
// Use perturb and deperturb to move forward and rewind.
// Untill https://github.com/tensorflow/tensorflow/issues/16464 is resolved, you must pull on incGen and decGen _after_ pulling on perturb or deperturb.
func NewSeedSM(s *op.Scope, noise NoiseFunc, paramDefs []ParamDef) (makeSeedSM func(*tf.Session) (SeedSM, error), generation tf.Output, params []tf.Output) {
	gen := op.Placeholder(s.SubScope("gen"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	seed := op.Placeholder(s.SubScope("seed"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	generationScope := s.SubScope("generation")
	generationVar := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))
	updateGeneration := op.AssignVariableOp(generationScope, generationVar, gen)
	initGeneration := op.AssignVariableOp(generationScope.SubScope("init"), generationVar, op.Const(generationScope.SubScope("zero"), int64(0)))
	generation = op.ReadVariableOp(generationScope, generationVar, tf.Int64)

	// Now we create slices to hold various things for each param.
	paramCount := len(paramDefs)                    // number of params
	varHandles := make([]tf.Output, paramCount)     // handles to the actual variables
	params = make([]tf.Output, paramCount)          // outputs to read the value of the params
	initParams := make([]*tf.Operation, paramCount) // operations to initialise the variables with zeros
	perturb := make([]*tf.Operation, paramCount)    // for perturbing according to the given seed.
	deperturb := make([]*tf.Operation, paramCount)  // for deperturbing according to the seed poped off the stack.
	for i, pd := range paramDefs {                  // for each tensor of params,
		paramScope := s.SubScope(pd.Name)
		paramIndex := op.Const(s.SubScope("param_index"), int64(i)) // the index of that param.
		zeroParam := pd.Init(paramScope.SubScope("init_val"))
		varHandles[i] = op.VarHandleOp(paramScope, zeroParam.DataType(), zeroParam.Shape(), op.VarHandleOpSharedName(pd.Name))
		initParams[i] = op.AssignVariableOp(paramScope, varHandles[i], zeroParam)      // OPs to initialize the param tensors.
		params[i] = op.ReadVariableOp(paramScope, varHandles[i], zeroParam.DataType()) // OPs to read them
		paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
		seedNoise := noise(paramScope.SubScope("perturb_noise"), op.Shape(paramScope, zeroParam), paramSeed, gen)
		//seedNoise = op.Print(paramScope, seedNoise, []tf.Output{seedNoise, gen, seed}, op.PrintMessage("sm"))
		perturb[i] = op.AssignAddVariableOp(paramScope.SubScope("perturb"), varHandles[i], seedNoise)
		deperturb[i] = op.AssignSubVariableOp(paramScope.SubScope("deperturb"), varHandles[i], seedNoise)
	}
	makeSeedSM = func(sess *tf.Session) (sm SeedSM, err error) {
		_, err = sess.Run(nil, nil, append(initParams, initGeneration))
		sm = SeedSM{
			sess:             sess,
			perturb:          perturb,
			deperturb:        deperturb,
			seedPH:           seed,
			genPH:            gen,
			updateGeneration: updateGeneration,
		}
		return
	}
	return
}

// NewBestSeed creates the OPs to search for good seeds from the current params.
func NewBestSeed(s *op.Scope,
	params []tf.Output,
	lossFunc LossFunc,
	noise NoiseFunc,
	numSeeds int,
	generation tf.Output,
) (
	makeBestSeed func(*tf.Session) (func() (int64, error), error),
) {
	// Now we create slices to hold various things for each param.
	paramCount := len(params) // number of params
	seedLosses := make([]tf.Output, numSeeds)
	one := op.Const(s.SubScope("one"), int64(1))
	// for each seed,
	for seedIndex := 0; seedIndex < numSeeds; seedIndex++ {
		seedScope := s.SubScope("child" + strconv.Itoa(seedIndex))
		seed := op.Const(seedScope.SubScope("seed"), int64(seedIndex))
		perturbedParams := make([]tf.Output, paramCount)
		// for each param,
		for i, param := range params {
			paramScope := seedScope.SubScope("param_" + strconv.Itoa(i))
			paramShape := op.Shape(paramScope.SubScope("input"), param, op.ShapeOutType(tf.Int32))
			paramIndex := op.Const(paramScope.SubScope("param_index"), int64(i))
			paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
			seedNoise := noise(paramScope.SubScope("perturb_noise"), paramShape, paramSeed, op.Add(paramScope.SubScope("inc_gen"), generation, one))
			perturbedParam := op.Add(paramScope, params[i], seedNoise)
			//perturbedParam = op.Print(paramScope, seedNoise, []tf.Output{seed, paramSeed, generation, perturbedParam}, op.PrintMessage("params"))
			perturbedParams[i] = perturbedParam
		}
		seedLosses[seedIndex] = lossFunc(seedScope.SubScope("model"), perturbedParams)
	}
	losses := op.Pack(s.SubScope("pack"), seedLosses)
	//losses = op.Print(s, losses, []tf.Output{losses}, op.PrintMessage("losses"), op.PrintSummarize(10))
	lowestSeed := op.ArgMin(s, losses, op.Const(s.SubScope("argmin_dims"), int32(0)), op.ArgMinOutputType(tf.Int64))
	makeBestSeed = func(sess *tf.Session) (bestSeed func() (int64, error), err error) {
		bestSeed = func() (seed int64, err error) {
			results, err := sess.Run(nil, []tf.Output{lowestSeed}, nil)
			if err != nil {
				return
			}
			seed = results[0].Value().(int64)
			return
		}
		return
	}
	return
}

// WeightedSeedSM allows one to move through the parameter space using a list of seed weights.
type WeightedSeedSM struct {
	Generation       int64
	SeedWeights      [][]float32
	perturb          []*tf.Operation
	deperturb        []*tf.Operation
	sess             *tf.Session
	seedPH           tf.Output
	genPH            tf.Output
	updateGeneration *tf.Operation
}

// NewWeightedSeedSM creates a state machine that moves through parameter space via a list of seed weights
func NewWeightedSeedSM(s *op.Scope, noise NoiseFunc, paramDefs []ParamDef) (makeSeedSM func(*tf.Session) (WeightedSeedSM, error), generation tf.Output, params []tf.Output) {
	gen := op.Placeholder(s.SubScope("gen"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	seed := op.Placeholder(s.SubScope("seed"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	generationScope := s.SubScope("generation")
	generationVar := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))
	updateGeneration := op.AssignVariableOp(generationScope, generationVar, gen)
	initGeneration := op.AssignVariableOp(generationScope.SubScope("init"), generationVar, op.Const(generationScope.SubScope("zero"), int64(0)))
	generation = op.ReadVariableOp(generationScope, generationVar, tf.Int64)

	// Now we create slices to hold various things for each param.
	paramCount := len(paramDefs)                    // number of params
	varHandles := make([]tf.Output, paramCount)     // handles to the actual variables
	params = make([]tf.Output, paramCount)          // outputs to read the value of the params
	initParams := make([]*tf.Operation, paramCount) // operations to initialise the variables with zeros
	perturb := make([]*tf.Operation, paramCount)    // for perturbing according to the given seed.
	deperturb := make([]*tf.Operation, paramCount)  // for deperturbing according to the seed poped off the stack.
	for i, pd := range paramDefs {                  // for each tensor of params,
		paramScope := s.SubScope(pd.Name)
		paramIndex := op.Const(s.SubScope("param_index"), int64(i)) // the index of that param.
		zeroParam := pd.Init(paramScope.SubScope("init_val"))
		varHandles[i] = op.VarHandleOp(paramScope, zeroParam.DataType(), zeroParam.Shape(), op.VarHandleOpSharedName(pd.Name))
		initParams[i] = op.AssignVariableOp(paramScope, varHandles[i], zeroParam)      // OPs to initialize the param tensors.
		params[i] = op.ReadVariableOp(paramScope, varHandles[i], zeroParam.DataType()) // OPs to read them
		paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
		seedNoise := noise(paramScope.SubScope("perturb_noise"), op.Shape(paramScope, zeroParam), paramSeed, gen)
		//seedNoise = op.Print(paramScope, seedNoise, []tf.Output{seedNoise, gen, seed}, op.PrintMessage("sm"))
		perturb[i] = op.AssignAddVariableOp(paramScope.SubScope("perturb"), varHandles[i], seedNoise)
		deperturb[i] = op.AssignSubVariableOp(paramScope.SubScope("deperturb"), varHandles[i], seedNoise)
	}
	makeSeedSM = func(sess *tf.Session) (sm WeightedSeedSM, err error) {
		_, err = sess.Run(nil, nil, append(initParams, initGeneration))
		sm = WeightedSeedSM{
			sess:             sess,
			perturb:          perturb,
			deperturb:        deperturb,
			seedPH:           seed,
			genPH:            gen,
			updateGeneration: updateGeneration,
		}
		return
	}
	return
}
