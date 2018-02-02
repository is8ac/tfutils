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
// Use perturb and deperturb to move forward or rewind.
func NewSeedSM(s *op.Scope,
	noise NoiseFunc,
	paramDefs []ParamDef,
	numSeeds int,
) (
	makeSeedSM func(*tf.Session) (SeedSM, error),
	newBestSeed func(LossFunc) func(*tf.Session) (func() (int64, error), error),
	generation tf.Output,
	params []tf.Output,
) {
	// we make two place holders for the go code to pass the seed, and the generation at run time.
	seed := op.Placeholder(s.SubScope("seed"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	gen := op.Placeholder(s.SubScope("gen"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	// also store the generation in a tf variable so that bestSeed can read it.
	generationScope := s.SubScope("generation")
	generationVar := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))
	updateGeneration := op.AssignVariableOp(generationScope, generationVar, gen)
	initGeneration := op.AssignVariableOp(generationScope.SubScope("init"), generationVar, op.Const(generationScope.SubScope("zero"), int64(0)))
	generation = op.ReadVariableOp(generationScope, generationVar, tf.Int64)

	paramCount := len(paramDefs) // number of params
	// Now we create slices to hold various things for each param.
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
		// we sum the seed with the index of the parameter tensor as a hack to prevent two parameter tensors of the same shape from being the same.
		paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
		seedNoise := noise(paramScope.SubScope("perturb_noise"), op.Shape(paramScope, zeroParam), paramSeed, gen)
		perturb[i] = op.AssignAddVariableOp(paramScope.SubScope("perturb"), varHandles[i], seedNoise)
		deperturb[i] = op.AssignSubVariableOp(paramScope.SubScope("deperturb"), varHandles[i], seedNoise)
	}
	// Once the user finalizes the scope and gives us session we can run the initialization OPs and construct the state SeedSM struct.
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
	// If the user also wants to search for the next best seed, they can run this func.
	newBestSeed = func(lossFunc LossFunc) (makeBestSeed func(*tf.Session) (func() (int64, error), error)) {
		bestSeedScope := s.SubScope("best_seed")
		paramCount := len(params) // number of params
		seedLosses := make([]tf.Output, numSeeds)
		one := op.Const(bestSeedScope.SubScope("one"), int64(1)) // needed later
		// for each seed,
		for seedIndex := 0; seedIndex < numSeeds; seedIndex++ {
			seedScope := bestSeedScope.SubScope("child" + strconv.Itoa(seedIndex))
			seed := op.Const(seedScope.SubScope("seed"), int64(seedIndex))
			perturbedParams := make([]tf.Output, paramCount)
			// for each param,
			for i, param := range params {
				paramScope := seedScope.SubScope("param_" + strconv.Itoa(i))
				paramShape := op.Shape(paramScope.SubScope("input"), param, op.ShapeOutType(tf.Int32))
				paramIndex := op.Const(paramScope.SubScope("param_index"), int64(i))
				paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
				seedNoise := noise(paramScope.SubScope("perturb_noise"),
					paramShape,
					paramSeed,
					op.Add(paramScope.SubScope("inc_gen"), generation, one), // this is a hack to get the generation to be correct.
				)
				perturbedParam := op.Add(paramScope, params[i], seedNoise)
				perturbedParams[i] = perturbedParam
			}
			seedLosses[seedIndex] = lossFunc(seedScope.SubScope("model"), perturbedParams)
		}
		losses := op.Pack(bestSeedScope.SubScope("pack"), seedLosses)
		lowestSeed := op.ArgMin(bestSeedScope, losses, op.Const(bestSeedScope.SubScope("argmin_dims"), int32(0)), op.ArgMinOutputType(tf.Int64))
		// once the user has given us the session, we can make the bestSeed func.
		makeBestSeed = func(sess *tf.Session) (bestSeed func() (int64, error), err error) {
			// Nothing needs to be finalized this time.
			bestSeed = func() (seed int64, err error) { // each time the user calls bestSeed(),
				results, err := sess.Run(nil, []tf.Output{lowestSeed}, nil) // pull on lowestSeed,
				if err != nil {
					return
				}
				seed = results[0].Value().(int64) // and give them the resulting index.
				return
			}
			return
		}
		return
	}
	return
}

// Step moves the parameters through parameter space by one list of weights
func (sm *WeightedSeedSM) Step(weights []float32) (err error) {
	sm.Generation++
	sm.SeedWeights = append(sm.SeedWeights, weights)
	weightsTensor, err := tf.NewTensor(weights)
	if err != nil {
		panic(err)
	}
	genTensor, err := tf.NewTensor(sm.Generation)
	if err != nil {
		panic(err)
	}
	_, err = sm.sess.Run(map[tf.Output]*tf.Tensor{sm.weightsPH: weightsTensor, sm.genPH: genTensor}, nil, append(sm.perturb, sm.updateGeneration))
	if err != nil {
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
	weightsPH        tf.Output
	genPH            tf.Output
	updateGeneration *tf.Operation
}

// NewWeightedSeedSM creates TF OPs for a state machine to move through parameter space according to the seed which is give and the generation.
// Use perturb and deperturb to move forward and rewind.
func NewWeightedSeedSM(s *op.Scope,
	noise NoiseFunc,
	paramDefs []ParamDef,
	numSeeds int,
) (
	makeSeedSM func(*tf.Session) (WeightedSeedSM, error),
	newSeedWeights func(LossFunc, tf.Output) func(*tf.Session) (func() ([]float32, error), error),
	generation tf.Output,
	params []tf.Output,
) {
	// we make two placeholders for the go code to pass the seed, and the generation at run time.
	weights := op.Placeholder(s.SubScope("seed"), tf.Float, op.PlaceholderShape(tf.MakeShape(int64(numSeeds))))
	gen := op.Placeholder(s.SubScope("gen"), tf.Int64, op.PlaceholderShape(tf.ScalarShape()))
	// also store the generation in a tf variable so that bestSeed can read it.
	generationScope := s.SubScope("generation")
	generationVar := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))
	updateGeneration := op.AssignVariableOp(generationScope, generationVar, gen)
	initGeneration := op.AssignVariableOp(generationScope.SubScope("init"), generationVar, op.Const(generationScope.SubScope("zero"), int64(0)))
	generation = op.ReadVariableOp(generationScope, generationVar, tf.Int64)

	paramCount := len(paramDefs) // number of params
	// Now we create slices to hold various things for each param.
	varHandles := make([]tf.Output, paramCount)     // handles to the actual variables
	params = make([]tf.Output, paramCount)          // outputs to read the value of the params
	initParams := make([]*tf.Operation, paramCount) // operations to initialise the variables with zeros
	perturb := make([]*tf.Operation, paramCount)    // for perturbing according to the given seed.
	deperturb := make([]*tf.Operation, paramCount)  // for deperturbing according to the seed poped off the stack.
	seedWeights := op.Unpack(s, weights, int64(numSeeds))
	for i, pd := range paramDefs { // for each tensor of params,
		paramScope := s.SubScope(pd.Name)
		zeroParam := pd.Init(paramScope.SubScope("init_val"))
		paramShape := op.Shape(paramScope, zeroParam)
		varHandles[i] = op.VarHandleOp(paramScope, zeroParam.DataType(), zeroParam.Shape(), op.VarHandleOpSharedName(pd.Name))
		initParams[i] = op.AssignVariableOp(paramScope, varHandles[i], zeroParam)      // OPs to initialize the param tensors.
		params[i] = op.ReadVariableOp(paramScope, varHandles[i], zeroParam.DataType()) // OPs to read them
		noises := make([]tf.Output, numSeeds)
		for s := range noises {
			seedScope := paramScope.SubScope("seed_" + strconv.Itoa(s))
			// we sum the seed with the index of the parameter tensor as a hack to prevent two parameter tensors of the same shape from being the same.
			seed := op.Const(seedScope, int64(s+i))
			seedNoise := noise(seedScope.SubScope("noise"), paramShape, seed, gen)
			noises[s] = op.Mul(seedScope, seedNoise, seedWeights[s])
		}
		weightedNoises := op.AddN(paramScope, noises)
		perturb[i] = op.AssignAddVariableOp(paramScope.SubScope("perturb"), varHandles[i], weightedNoises)
	}
	// Once the user finalizes the scope and gives us session we can run the initialization OPs and construct the state SeedSM struct.
	makeSeedSM = func(sess *tf.Session) (sm WeightedSeedSM, err error) {
		_, err = sess.Run(nil, nil, append(initParams, initGeneration))
		sm = WeightedSeedSM{
			sess:             sess,
			perturb:          perturb,
			deperturb:        deperturb,
			weightsPH:        weights,
			genPH:            gen,
			updateGeneration: updateGeneration,
		}
		return
	}
	// If the user also wants to calculate the weights, they can run this func.
	newSeedWeights = func(lossFunc LossFunc, seedWeight tf.Output) (makeSeedWeights func(*tf.Session) (func() ([]float32, error), error)) {
		seedWeightsScope := s.SubScope("seed_weights")
		paramCount := len(params) // number of params
		seedLosses := make([]tf.Output, numSeeds)
		one := op.Const(seedWeightsScope.SubScope("one"), int64(1))        // needed later
		curLoss := lossFunc(seedWeightsScope.SubScope("cur_loss"), params) // the loss if the params are unperturbed.
		seedDeltas := make([]tf.Output, numSeeds)                          // how much did each seed change the loss?
		// for each seed,
		for s := range seedDeltas {
			seedScope := seedWeightsScope.SubScope("child" + strconv.Itoa(s))
			seed := op.Const(seedScope.SubScope("seed"), int64(s))
			perturbedParams := make([]tf.Output, paramCount)
			// perturb all the params
			for i, param := range params {
				paramScope := seedScope.SubScope("param_" + strconv.Itoa(i))
				paramShape := op.Shape(paramScope.SubScope("input"), param, op.ShapeOutType(tf.Int32))
				paramIndex := op.Const(paramScope.SubScope("param_index"), int64(i))
				paramSeed := op.Add(paramScope.SubScope("inc_seed"), seed, paramIndex)
				seedNoise := noise(paramScope.SubScope("perturb_noise"),
					paramShape,
					paramSeed,
					op.Add(paramScope.SubScope("inc_gen"), generation, one), // this is a hack to get the generation to be correct.
				)
				perturbedParam := op.Add(paramScope, params[i], seedNoise)
				perturbedParams[i] = perturbedParam
			}
			seedLosses[s] = lossFunc(seedScope.SubScope("model"), perturbedParams)
		}
		losses := op.Pack(seedWeightsScope.SubScope("pack"), seedLosses)
		weights := op.Mul(s, op.Sub(s, curLoss, losses), seedWeight)
		// once the user has given us the session, we can make the bestSeed func.
		makeSeedWeights = func(sess *tf.Session) (seedWeights func() ([]float32, error), err error) {
			// Nothing needs to be finalized this time.
			seedWeights = func() (weightsVals []float32, err error) { // each time the user calls bestWeights(),
				results, err := sess.Run(nil, []tf.Output{weights}, nil) // pull on lowestSeed,
				if err != nil {
					return
				}
				weightsVals = results[0].Value().([]float32) // and give them the resulting weights.
				return
			}
			return
		}
		return
	}
	return
}
