-- neuralMario by Maxiwell Luo, Srisri Soundararajan, and Shahzar Rizvi
-- For the fourth quarter project of Computer Science Topics 2015-2016 taught by Dr. Lopykinski

--[[ Until the ending comment, the following code was copied from SethBling's MarI/O.
		This code was copied because retrieving inputs from the emulator is out of the scope of this project.--]]
Filename = "save.state";
ButtonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"};

BoxRadius = 6;
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1);

Inputs = InputSize+1;
Outputs = #ButtonNames;

Population = 300;
DeltaDisjoint = 2.0;
DeltaWeights = 0.4;
DeltaThreshold = 1.0;

StaleSpecies = 15;

MutateConnectionsChance = 0.25;
PerturbChance = 0.90;
CrossoverChance = 0.75;
LinkMutationChance = 2.0;
NodeMutationChance = 0.50;
BiasMutationChance = 0.40;
StepSize = 0.1;
DisableMutationChance = 0.4;
EnableMutationChance = 0.2;

TimeoutConstant = 20;

MaxNodes = 1000000;

-- Finds the positions of things on the screen at one moment??
function getPositions()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                marioX = memory.read_s16_le(0x94);
                marioY = memory.read_s16_le(0x96);

                local layer1x = memory.read_s16_le(0x1A);
                local layer1y = memory.read_s16_le(0x1C);

                screenX = marioX-layer1x;
                screenY = marioY-layer1y;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86);
                marioY = memory.readbyte(0x03B8)+16;

                screenX = memory.readbyte(0x03AD);
                screenY = memory.readbyte(0x03B8);
        end
end

function getTile(dx, dy)
        if gameinfo.getromname() == "Super Mario World (USA)" then
                x = math.floor((marioX+dx+8)/16);
                y = math.floor((marioY+dy)/16);

                return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10);
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local x = marioX + dx + 8;
                local y = marioY + dy - 16;
                local page = math.floor(x/256)%2;

                local subx = math.floor((x%256)/16);
                local suby = math.floor((y - 32)/16);
                local addr = 0x500 + page*13*16+suby*16+subx;

                if suby >= 13 or suby < 0 then
                        return 0;
                end

                if memory.readbyte(addr) ~= 0 then
                        return 1;
                else
                        return 0;
                end
        end
end

function getSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local sprites = {};
                for slot=0,11 do
                        local status = memory.readbyte(0x14C8+slot);
                        if status ~= 0 then
                                spritex = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256;
                                spritey = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256;
                                sprites[#sprites+1] = {["x"]=spritex, ["y"]=spritey};
                        end
                end

                return sprites;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local sprites = {};
                for slot=0,4 do
                        local enemy = memory.readbyte(0xF+slot);
                        if enemy ~= 0 then
                                local ex = memory.readbyte(0x6E + slot)*0x100 + memory.readbyte(0x87+slot);
                                local ey = memory.readbyte(0xCF + slot)+24;
                                sprites[#sprites+1] = {["x"]=ex,["y"]=ey};
                        end
                end

                return sprites;
        end
end

function getExtendedSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local extended = {};
                for slot=0,11 do
                        local number = memory.readbyte(0x170B+slot);
                        if number ~= 0 then
                                spritex = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256;
                                spritey = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256;
                                extended[#extended+1] = {["x"]=spritex, ["y"]=spritey};
                        end
                end

                return extended;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                return {};
        end
end

function getInputs()
        getPositions();

        sprites = getSprites();
        extended = getExtendedSprites();

        local inputs = {};

        for dy=-BoxRadius*16,BoxRadius*16,16 do
                for dx=-BoxRadius*16,BoxRadius*16,16 do
                        inputs[#inputs+1] = 0;

                        tile = getTile(dx, dy);
                        if tile == 1 and marioY+dy < 0x1B0 then
                                inputs[#inputs] = 1;
                        end

                        for i = 1,#sprites do
                                distx = math.abs(sprites[i]["x"] - (marioX+dx));
                                disty = math.abs(sprites[i]["y"] - (marioY+dy));
                                if distx <= 8 and disty <= 8 then
                                        inputs[#inputs] = -1;
                                end
                        end

                        for i = 1,#extended do
                                distx = math.abs(extended[i]["x"] - (marioX+dx));
                                disty = math.abs(extended[i]["y"] - (marioY+dy));
                                if distx < 8 and disty < 8 then
                                        inputs[#inputs] = -1;
                                end
                        end
                end
        end

        --mariovx = memory.read_s8(0x7B)
        --mariovy = memory.read_s8(0x7D)

        return inputs;
end

--[[ The code above is copied from SethBling's MarI/O --]]

function sigmoid(x)
	return 1 / (1 + math.exp(-4.9 * x));
end

function newNeuron()
	local neuron = {};
	neuron.id = 0;
	neuron.neuronType = -1;
	-- input = 0, hidden = 1, output = 2, bias = 3
	neuron.isRecurrent = false;
	neuron.activationResponse = 0.0;
	neuron.splitX = 0.0;
	neuron.splitY = 0.0;
	neuron.incoming = {};
	return neuron;
end

function newLink()
	local link = {};
	link.enter = 0;
	link.out = 0;
	link.weight = 0.0;
	link.id = 0;
	link.enabled = true;
	return link;
end

function copyLink(toCopy)
	local link = newLink();
	link.enter = toCopy.enter;
	link.out = toCopy.out;
	link.weight = toCopy.weight;
	link.id = toCopy.id;
	link.enabled = toCopy.enabled;
	return link;
end

function newGenome()
	local genome = {};
	genome.neurons = {};
	genome.links = {};
	genome.network = {};
	genome.maxneuron = 0;
	genome.rawFitness = 0.0;
	genome.adjustedFitness = 0.0;
	genome.amountToSpawn = 0.0;
	genome.numInputs = 0;
	genome.numOutputs = 0;
	genome.species = 0;
	genome.mutationRates = {};
	genome.mutationRates["connections"] = MutateConnectionsChance;
	genome.mutationRates["link"] = LinkMutationChance;
	genome.mutationRates["bias"] = BiasMutationChance;
	genome.mutationRates["node"] = NodeMutationChance;
	genome.mutationRates["enable"] = EnableMutationChance;
	genome.mutationRates["disable"] = DisableMutationChance;
	genome.mutationRates["step"] = StepSize;
	genome.globalRank = 0;
	return genome;
end

function copyGenome(genome)
	local copy = newGenome();

	for i = 1, #genome.links do
		table.insert(copy.links, copyLink(genome.links[i]));
	end

	copy.maxneuron = genome.maxneuron;

	for mutation, rate in pairs(genome.mutationRates) do
		copy.mutationRates[mutation] = rate;
	end

	return copy;
end

function newSpecies()
	local species = {};
	species.genomes{};
	species.topFitness = 0;
	species.avgFitness = 0;
	species.staleness = 0;
	return species;
end

function newPool()
	local pool = {};
	pool.species = {};
	pool.generation = 0;
	pool.innovation = Ouuputs;
	pool.currentSpecies = 1;
	pool.currentGenome = 1;
	pool.currentFrame = 0;
	pool.maxFitness = 0;
	return pool;
end

function initializePool()
	pool = newPool(); -- the entire population (all generations)

end

function createNetwork(genome)
	local network = {};
	network.neurons = {};
	-- The inputs from the screen
	for i = 1, Inputs do
		network.neuron[i] = newNeuron();
	end

	-- The buttons (Outputs)
	for i = 1, Outputs do
		network.neuron[MaxNodes + i] = newNeuron();
	end

	table.sort(genome.links, function(a, b) return a.out < b.out; end);

	-- Hidden nodes
	for i = 1, #genomes.genes do
		local link = genome.links[i];
		if link.enabled then
			if network.neurons[link.out] == nil then
				network.neurons[link.out] = newNeuron();
			end
			table.insert(network.neurons[link.out].incoming, link);
			if network.neurons[link.enter] == nil then
				network.neurons[link.enter] = newNeuron();
			end
		end
	end

	genome.network = network;

end

function evaluateNetwork(network, inputs)
	table.insert(inputs, 1);
	if #inputs ~= Inputs then
		console.writeline("Inputs don't match");
	end

	for i = 1, #network.neurons do
		local sum = 0;
		for j = 1, #network.neurons[i].incoming do
			sum = sum + network.neurons[i].incoming[j].weight * network.neurons[i].incoming[j].enter.value;
		end

		if #network.neurons[i].incoming > 0 then
			network.neurons[i].value = sigmoid(sum);
		end
	end

	local outputs = {};
	for i = 1, Outputs do
		if network.neurons[MaxNodes + i].value > .5 then
			outputs[ButtonNames[i]] = true;
		else
			outputs[ButtonNames[i]] = false;
		end
	end

	return outputs;
end

function weightMutate(genome)
	for i = 1, #genome.links do
		if math.random() < PerturbChance then
			genome.link[i] = genome.link[i] + math.random() * step * 2 - step;
		else
			genome.link[i] = math.random() * 4 - 2;
		end
	end
end

function containsLink(links, link)
	for i = 1, #links do
		if links[i].enter == link.enter and links[i].out == link.out then
			return true;
		end
	end
	return false;
end

function randomNeuron(links, includeInputs)
	local neurons = {};
	if includeInputs then
		for i = 1, Inputs do
			neurons[i] = true;
		end
	end

	for i = 1, Outputs do
		neurons[MaxNodes + i] = true;
	end

	for i = 1, #links do
		if links[i].enter > Inputs then
			neurons[links[i].enter] = true;
		end
		if links[i].out > Inputs then
			neurons[links[i].out] = true;
		end
	end

	local count = 0;

	for _, _ in pairs(neurons) do
		count = count + 1;
	end

	count = random(1, count);

	for key, value in pairs(neurons) do
		count = count - 1;
		if count == 0 then
			return key;
		end
	end
	return 0;
end

function newInnovation()
	pool.innovation = pool.innovation + 1;
	return pool.innovation;
end

function addLinkMutate(genome, forceBias)
	local neuron1 = randomNeuron(genome.links, true);
	local neuron2 = randomNeuron(genome.links, false);

	local nLink = newLink();
	nLink.enter = neuron1;
	nLink.out = neuron2;

	if forceBias then
		nLink.enter = Inputs; -- Set it to the bias node
	end

	if containsLink(genome.links, link) then
		return;
	end

	nLink.id = newInnovation();

	nLink.weight = math.random() * 4 - 2;
	table.insert(genome.links, nLink);
end

function addNeuronMutate(genome)
	if #genome.links == 0 then
		return -- nowhere to add a node
	end

	genome.maxneuron = genome.maxneuron + 1;

	local replace = genome.links[math.random(#genome.links)];
	if not replace.enabled then
		return
	end
	replace.enabled = false;

	local nLink1 = copyLink(replace);
	nLink1.out = neuron.maxneuron;
	nLink1.weight = 1.0;
	nLink1.enabled = true;
	nLink1.innovation = newInnovation();
	table.insert(genome.links, nLink1);

	local nLink2 = copyLink(replace);
	nLink2.enter = neuron.maxneuron;
	nLink2.enabled = true;
	nLink2.innovation = newInnovation();
	table.insert(genome.links, nLink2);
end

function toggleEnableMutate(genome, enable)
	local toToggle = {};
	for _, gene in pairs(genome.links) do
		if gene.enabled == enable then
			table.insert(toToggle, gene)
		end
	end

	if #toToggle == 0 then
		return
	end

	local gene = toToggle[math.random(#toToggle)];
	gene.enabled = not gene.enabled;
end

function mutate(genome)
	--Change mutation rates
	for mutation, rate in pairs(genome.mutationRates) do
		if math.random(2) == 1 then
			genome[mutation] = .95 * rate; -- 95/100
		else
			genome[mutation] = 1.05263 * rate; -- 100/95
		end
	end

	if math.random() < genome.mutationRates["connections"] then
		pointMutate(genome);
	end

	local p = genome.mutationRates["link"];
	while p > 0 do
		if math.random() < p then
			addLinkMutate(genome, false);
		end
		p = p - 1;
	end

	p = genome.mutationRates["bias"];
	while p > 0 do
		if math.random() < p then
			addLinkMutate(genome, true);
		end
		p = p - 1;
	end

	p = genome.mutationRates["node"];
	while p > 0 do
		if math.random() < p then
			addNeuronMutate(genome);
		end
		p = p - 1;
	end

	p = genome.mutationRates["enable"];
	while p > 0 do
		if math.random() < p then
			toggleEnableMutate(genome, true);
		end
		p = p - 1;
	end

	p = genome.mutationRates["disable"];
	while p > 0 do
		if math.random() < p then
			toggleEnableMutate(genome, false);
		end
		p = p - 1;
	end
end

function crossover(genome1, genome2)
	if genome1.fitness < genome2.fitness then
		genome1, genome2 = genome2, genome1;
	end

	local linkIds2 = {};
	for i = 1, #genome2.links do
		linkIds2[genome2.links[i].id] = genome2.links[i];
	end

	local child = newGenome();

	for i = 1, #genome1.genes do
		if linksIds2[genome1.genes[i].id] ~= nil and math.random(2) == 1 and linksIds2[genome1.genes[i].id].enabled == true then
			table.insert(child.genes, linksIds2[genome1.genes[i].id]);
		else
			table.insert(child.genes, genome1.genes[i]);
		end
	end

	child.maxneuron = math.max(genome1.maxneuron, genome2.maxneuron);

	for mutation, rate in pairs(genome1.mutationRates) do
		child.mutationRates[mutation] = rate;
	end

	return child;
end

function breedChild(species)
	local child = {};
	if math.random() < CrossoverChance then
		child = crossover(species.genomes[math.random(#species.genomes)], species.genomes[math.random(#species.genomes)]);
	else
		child = copyGenome(species.genomes[math.random(#species.genomes)]);
	end
	mutate(child);
	return child;
end

function disjointExcess(links1, links2)
	local linkIds1 = {};
	for i = 1, #links1 do
		linkIds1[links1[i].id] = true;
	end

	local linkIds2 = {};
	for i = 1, #links2 do
		linkIds2[links1[i].id] = true;
	end

	local disjointExcessLinks = 0;
	for i = 1, #links1 do
		if not linkIds2[links1[i].id] then
			disjointExcessLinks = disjointExcessLinks + 1;
		end
	end

	for i = 1, #links2 do
		if not linkIds1[links2[i].id] then
			disjointExcessLinks = disjointExcessLinks + 1;
		end
	end

	return disjointExcessGenes / math.max(#links1, #links2);
end

function avgWeightDifference(links1, links2)
	local linkIds2 = {};
	for i = 1, #links2 do
		linkIds2[links2[i].id] = links2[i];
	end

	local difference = 0.0;
	local count = 0;
	for i = 1, #links1 do
		if linkIds2[links1[i].id] ~= nil then
			difference = difference + math.abs(link1[i].weight - linkIds2[link1[i].id].weight);
			count = count + 1;
		end
	end

	return difference / count;
end

function sameSpecies(genome1, genome2)
	return DeltaDisjoint * disjointExcess(genome1.links, genome2.links) + DeltaWeights * avgWeightDifference(genome1.links, genome2.links) < DeltaThreshold;
end
