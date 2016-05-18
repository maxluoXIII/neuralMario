-- neuralMario by Maxiwell Luo, Sriram Soundararajan, and Shahzar Rizvi
-- For the fourth quarter project of Computer Science Topics 2015-2016 taught by Dr. Lopykinski

--[[ Until the ending comment, the following code was copied from SethBling's MarI/O.
		This code was copied because retrieving inputs from the emulator is out of the scope of this project.--]]
Filename = "Super Mario World (USA).Performance.QuickSave0.state";
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
                marioX = memory.read_s16_le(0x94)
                marioY = memory.read_s16_le(0x96)

                local layer1x = memory.read_s16_le(0x1A);
                local layer1y = memory.read_s16_le(0x1C);

                screenX = marioX-layer1x
                screenY = marioY-layer1y
        elseif gameinfo.getromname() == "Super Mario Bros." then
                marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86)
                marioY = memory.readbyte(0x03B8)+16

                screenX = memory.readbyte(0x03AD)
                screenY = memory.readbyte(0x03B8)
        end
end

function getTile(dx, dy)
        if gameinfo.getromname() == "Super Mario World (USA)" then
                x = math.floor((marioX+dx+8)/16)
                y = math.floor((marioY+dy)/16)

                return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10)
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local x = marioX + dx + 8
                local y = marioY + dy - 16
                local page = math.floor(x/256)%2

                local subx = math.floor((x%256)/16)
                local suby = math.floor((y - 32)/16)
                local addr = 0x500 + page*13*16+suby*16+subx

                if suby >= 13 or suby < 0 then
                        return 0
                end

                if memory.readbyte(addr) ~= 0 then
                        return 1
                else
                        return 0
                end
        end
end

function getSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local sprites = {}
                for slot=0,11 do
                        local status = memory.readbyte(0x14C8+slot)
                        if status ~= 0 then
                                spritex = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256
                                spritey = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256
                                sprites[#sprites+1] = {["x"]=spritex, ["y"]=spritey}
                        end
                end

                return sprites
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local sprites = {}
                for slot=0,4 do
                        local enemy = memory.readbyte(0xF+slot)
                        if enemy ~= 0 then
                                local ex = memory.readbyte(0x6E + slot)*0x100 + memory.readbyte(0x87+slot)
                                local ey = memory.readbyte(0xCF + slot)+24
                                sprites[#sprites+1] = {["x"]=ex,["y"]=ey}
                        end
                end

                return sprites
        end
end

function getExtendedSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local extended = {}
                for slot=0,11 do
                        local number = memory.readbyte(0x170B+slot)
                        if number ~= 0 then
                                spritex = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256
                                spritey = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256
                                extended[#extended+1] = {["x"]=spritex, ["y"]=spritey}
                        end
                end

                return extended
        elseif gameinfo.getromname() == "Super Mario Bros." then
                return {}
        end
end

function getInputs()
        getPositions()

        sprites = getSprites()
        extended = getExtendedSprites()

        local inputs = {}

        for dy=-BoxRadius*16,BoxRadius*16,16 do
                for dx=-BoxRadius*16,BoxRadius*16,16 do
                        inputs[#inputs+1] = 0

                        tile = getTile(dx, dy)
                        if tile == 1 and marioY+dy < 0x1B0 then
                                inputs[#inputs] = 1
                        end

                        for i = 1,#sprites do
                                distx = math.abs(sprites[i]["x"] - (marioX+dx))
                                disty = math.abs(sprites[i]["y"] - (marioY+dy))
                                if distx <= 8 and disty <= 8 then
                                        inputs[#inputs] = -1
                                end
                        end

                        for i = 1,#extended do
                                distx = math.abs(extended[i]["x"] - (marioX+dx))
                                disty = math.abs(extended[i]["y"] - (marioY+dy))
                                if distx < 8 and disty < 8 then
                                        inputs[#inputs] = -1
                                end
                        end
                end
        end

        --mariovx = memory.read_s8(0x7B)
        --mariovy = memory.read_s8(0x7D)
        return inputs
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
	neuron.value = 0.0;
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
	genome.fitness = 0.0;
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
	species.genomes = {};
	species.topFitness = 0;
	species.averageRank = 0;
	species.staleness = 0;
	return species;
end

function newPool()
	local pool = {};
	pool.species = {};
	pool.generation = 0;
	pool.innovation = Outputs;
	pool.currentSpecies = 1;
	pool.currentGenome = 1;
	pool.currentFrame = 0;
	pool.maxFitness = 0;
	return pool;
end

function createNetwork(genome)
	local network = {};
	network.neurons = {};
	-- The inputs from the screen
	for i = 1, Inputs do
		network.neurons[i] = newNeuron();
	end

	-- The buttons (Outputs)
	for i = 1, Outputs do
		network.neurons[MaxNodes + i] = newNeuron();
	end

	table.sort(genome.links, function(a, b) return a.out < b.out; end);

	-- Hidden nodes
	for i = 1, #genome.links do
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

	for i=1,Inputs do
                network.neurons[i].value = inputs[i];
	end

	for _,neuron in pairs(network.neurons) do
		local sum = 0;
		for j = 1, #neuron.incoming do
			sum = sum + neuron.incoming[j].weight * network.neurons[neuron.incoming[j].enter].value;
		end

		if #neuron.incoming > 0 then
			neuron.value = sigmoid(sum);
		end
	end

	local outputs = {};
	for i = 1, Outputs do
		if network.neurons[MaxNodes + i].value > .5 then
			outputs["P1 " .. ButtonNames[i]] = true;
		else
			outputs["P1 " .. ButtonNames[i]] = false;
		end
	end

	return outputs;
end

function weightMutate(genome)
	local step = genome.mutationRates["step"];
	for i = 1, #genome.links do
		if math.random() < PerturbChance then
			genome.links[i].weight = genome.links[i].weight + math.random() * step * 2 - step;
		else
			genome.links[i].weight = math.random() * 4 - 2;
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

	count = math.random(count);

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

	if containsLink(genome.links, nLink) then
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
	nLink1.out = genome.maxneuron;
	nLink1.weight = 1.0;
	nLink1.enabled = true;
	nLink1.id = newInnovation();
	table.insert(genome.links, nLink1);

	local nLink2 = copyLink(replace);
	nLink2.enter = genome.maxneuron;
	nLink2.enabled = true;
	nLink2.id = newInnovation();
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
		weightMutate(genome);
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
	for i = 1, #genome1.links do
		if linkIds2[genome1.links[i].id] ~= nil and math.random(2) == 1 and linkIds2[genome1.links[i].id].enabled == true then
			table.insert(child.links, linkIds2[genome1.links[i].id]);
		else
			table.insert(child.links, genome1.links[i]);
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
		linkIds2[links2[i].id] = true;
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

	return disjointExcessLinks / math.max(#links1, #links2);
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
			difference = difference + math.abs(links1[i].weight - linkIds2[links1[i].id].weight);
			count = count + 1;
		end
	end

	return difference / count;
end

function sameSpecies(genome1, genome2)
	return DeltaDisjoint * disjointExcess(genome1.links, genome2.links) + DeltaWeights * avgWeightDifference(genome1.links, genome2.links) < DeltaThreshold;
end

function cullSpecies(onlyOne)
	for i = 1, #pool.species do
		local species = pool.species[i];

		table.sort(species.genomes, function(a, b)
			return a.fitness > b.fitness;
		end);

		local remaining = math.ceil(#species.genomes/2);
		if onlyOne then
			remaining = 1
		end

		while #species.genomes > remaining do
			table.remove(species.genomes);
		end
	end
end

function rankGlobally()
	local global = {};
	local numSpecies = #pool.species;
	for i = 1, numSpecies do
		local species = pool.species[i];
		for j = 1, #species.genomes do
			table.insert(global, species.genomes[j]);
		end
	end

	table.sort(global, function(a, b)
		return a.fitness < b.fitness;
	end);

	for i = 1, #global do
		global[i].globalRank = i;
	end
end

function removeStaleSpecies()
	local survivors = {};
	for i = 1, #pool.species do
		local species = pool.species[i]

		table.sort(species, function(a, b)
			return a.fitness > b.fitness;
		end);
		if species.genomes[1].fitness > species.topFitness then
			species.topFitness = species.genomes[1].fitness;
			species.staleness = 0;
		else
			species.staleness = species.staleness + 1;
		end
		if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness then
			table.insert(survivors, species);
		end
	end
	pool.species = survivors;
end

function totalAverageRank()
	local sum = 0;
	for i = 1, #pool.species do
		sum = sum + pool.species[i].averageRank;
	end
	return sum;
end


function removeWeakSpecies()
	local survivors = {};
	for i = 1, #pool.species do
		local species = pool.species[i];
		local total = totalAverageRank();
		breed = math.floor(species.averageRank/total * Population);
		if breed >= 1 then
			table.insert(survivors, species);
		end
	end
	pool.species = survivors;
end

function calcAverageRank(species)
	local sum = 0;
	for i = 1, #species.genomes do
		sum = sum + species.genomes[i].globalRank;
	end
	species.averageRank = sum/#species.genomes;
end

function basicGenome()
	local genome = newGenome();
	local innovation = 1;
	genome.maxneuron = Inputs;
	mutate(genome);
	return genome;
end

--copied Stuff
function writeFile(filename)
        local file = io.open(filename, "w")
        file:write(pool.generation .. "\n")
        file:write(pool.maxFitness .. "\n")
        file:write(#pool.species .. "\n")
        for n,species in pairs(pool.species) do
                file:write(species.topFitness .. "\n")
                file:write(species.staleness .. "\n")
                file:write(#species.genomes .. "\n")
                for m,genome in pairs(species.genomes) do
                        file:write(genome.fitness .. "\n")
                        file:write(genome.maxneuron .. "\n")
                        for mutation,rate in pairs(genome.mutationRates) do
                                file:write(mutation .. "\n")
                                file:write(rate .. "\n")
                        end
                        file:write("done\n")

                        file:write(#genome.links .. "\n")
                        for l,gene in pairs(genome.links) do
                                file:write(gene.enter .. " ")
                                file:write(gene.out .. " ")
                                file:write(gene.weight .. " ")
                                file:write(gene.id .. " ")
                                if(gene.enabled) then
                                        file:write("1\n")
                                else
                                        file:write("0\n")
                                end
                        end
                end
        end
        file:close()
end

function clearJoypad()
        controller = {};
        for b = 1,#ButtonNames do
                controller["P1 " .. ButtonNames[b]] = false;
        end
        joypad.set(controller);
end

function evaluateCurrent()
        local species = pool.species[pool.currentSpecies];
        local genome = species.genomes[pool.currentGenome];

        inputs = getInputs();
        controller = evaluateNetwork(genome.network, inputs);

        if controller["P1 Left"] and controller["P1 Right"] then
                controller["P1 Left"] = false;
                controller["P1 Right"] = false;
        end
        if controller["P1 Up"] and controller["P1 Down"] then
                controller["P1 Up"] = false;
                controller["P1 Down"] = false;
        end

        joypad.set(controller);
end

function initializeRun()
        savestate.load(Filename);
        rightmost = 0;
        pool.currentFrame = 0;
        timeout = TimeoutConstant;
        clearJoypad();

        local species = pool.species[pool.currentSpecies];
        local genome = species.genomes[pool.currentGenome];
        createNetwork(genome);
        evaluateCurrent();
end

function addToSpecies(child)
	local speciesFound = false;
	for i = 1, #pool.species do
		local species = pool.species[i];
		if not speciesFound and sameSpecies(child, species.genomes[1]) then
			table.insert(species.genomes, child);
			speciesFound = true;
		end
	end

	if not speciesFound then
		local newSpecies = newSpecies();
		table.insert(newSpecies.genomes, child);
		table.insert(pool.species, newSpecies);
	end
end

function initializePool()
	pool = newPool();

	for i = 1, Population do
		local basic = basicGenome();
		addToSpecies(basic);
	end

	initializeRun();
end



--Copied Stuffs

if pool == nil then
        initializePool();
end

function newGeneration()
	cullSpecies(false);
	rankGlobally();
	removeStaleSpecies();
	rankGlobally();
	for i = 1, #pool.species do
		calcAverageRank(pool.species[i]);
	end
	removeWeakSpecies();
	local children = {};
	for i = 1, #pool.species do
		local toBreed = math.floor(pool.species[i].averageRank / totalAverageRank() * Population) - 1;
		for j = 1, toBreed do
			table.insert(children, breedChild(pool.species[i]));
		end
	end
	cullSpecies(true);
	while #children + #pool.species < Population do
		local rand = pool.species[math.random(1, #pool.species)];
		table.insert(children, breedChild(rand));
	end
	for i = 1, #children do
		addToSpecies(children[i]);
	end

	pool.generation = pool.generation + 1;

	writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile));
end

function nextGenome()
        pool.currentGenome = pool.currentGenome + 1;
        if pool.currentGenome > #pool.species[pool.currentSpecies].genomes then
                pool.currentGenome = 1;
                pool.currentSpecies = pool.currentSpecies+1;
                if pool.currentSpecies > #pool.species then
                        newGeneration();
                        pool.currentSpecies = 1;
                end
        end
end

function fitnessAlreadyMeasured()
        local species = pool.species[pool.currentSpecies];
        local genome = species.genomes[pool.currentGenome];

        return genome.fitness ~= 0;
end

function displayGenome(genome)
        local network = genome.network
        local cells = {}
        local i = 1
        local cell = {}
        for dy=-BoxRadius,BoxRadius do
                for dx=-BoxRadius,BoxRadius do
                        cell = {}
                        cell.x = 50+5*dx
                        cell.y = 70+5*dy
                        cell.value = network.neurons[i].value
                        cells[i] = cell
                        i = i + 1
                end
        end
        local biasCell = {}
        biasCell.x = 80
        biasCell.y = 110
        biasCell.value = network.neurons[Inputs].value
        cells[Inputs] = biasCell

        for o = 1,Outputs do
                cell = {}
                cell.x = 220
                cell.y = 30 + 8 * o
                cell.value = network.neurons[MaxNodes + o].value
                cells[MaxNodes+o] = cell
                local color
                if cell.value > 0 then
                        color = 0xFF0000FF
                else
                        color = 0xFF000000
                end
                gui.drawText(223, 24+8*o, ButtonNames[o], color, 9)
        end

        for n,neuron in pairs(network.neurons) do
                cell = {}
                if n > Inputs and n <= MaxNodes then
                        cell.x = 140
                        cell.y = 40
                        cell.value = neuron.value
                        cells[n] = cell
                end
        end

        for n=1,4 do
                for _,gene in pairs(genome.links) do
                        if gene.enabled then
                                local c1 = cells[gene.enter]
                                local c2 = cells[gene.out]
                                if gene.enter > Inputs and gene.enter <= MaxNodes then
                                        c1.x = 0.75*c1.x + 0.25*c2.x
                                        if c1.x >= c2.x then
                                                c1.x = c1.x - 40
                                        end
                                        if c1.x < 90 then
                                                c1.x = 90
                                        end

                                        if c1.x > 220 then
                                                c1.x = 220
                                        end
                                        c1.y = 0.75*c1.y + 0.25*c2.y

                                end
                                if gene.out > Inputs and gene.out <= MaxNodes then
                                        c2.x = 0.25*c1.x + 0.75*c2.x
                                        if c1.x >= c2.x then
                                                c2.x = c2.x + 40
                                        end
                                        if c2.x < 90 then
                                                c2.x = 90
                                        end
                                        if c2.x > 220 then
                                                c2.x = 220
                                        end
                                        c2.y = 0.25*c1.y + 0.75*c2.y
                                end
                        end
                end
        end

        gui.drawBox(50-BoxRadius*5-3,70-BoxRadius*5-3,50+BoxRadius*5+2,70+BoxRadius*5+2,0xFF000000, 0x80808080)
        for n,cell in pairs(cells) do
                if n > Inputs or cell.value ~= 0 then
                        local color = math.floor((cell.value+1)/2*256)
                        if color > 255 then color = 255 end
                        if color < 0 then color = 0 end
                        local opacity = 0xFF000000
                        if cell.value == 0 then
                                opacity = 0x50000000
                        end
                        color = opacity + color*0x10000 + color*0x100 + color
                        gui.drawBox(cell.x-2,cell.y-2,cell.x+2,cell.y+2,opacity,color)
                end
        end
        for _,gene in pairs(genome.links) do
                if gene.enabled then
                        local c1 = cells[gene.enter]
                        local c2 = cells[gene.out]
                        local opacity = 0xA0000000
                        if c1.value == 0 then
                                opacity = 0x20000000
                        end

                        local color = 0x80-math.floor(math.abs(sigmoid(gene.weight))*0x80)
                        if gene.weight > 0 then
                                color = opacity + 0x8000 + 0x10000*color
                        else
                                color = opacity + 0x800000 + 0x100*color
                        end
                        gui.drawLine(c1.x+1, c1.y, c2.x-3, c2.y, color)
                end
        end

        gui.drawBox(49,71,51,78,0x00000000,0x80FF0000)

        if forms.ischecked(showMutationRates) then
                local pos = 100
                for mutation,rate in pairs(genome.mutationRates) do
                        gui.drawText(100, pos, mutation .. ": " .. rate, 0xFF000000, 10)
                        pos = pos + 8
                end
        end
end

function savePool()
        local filename = forms.gettext(saveLoadFile);
        writeFile(filename);
end

function loadFile(filename)
        local file = io.open(filename, "r");
        pool = newPool();
        pool.generation = file:read("*number");
        pool.maxFitness = file:read("*number");
        forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness));
        local numSpecies = file:read("*number");
        for s=1,numSpecies do
                local species = newSpecies();
                table.insert(pool.species, species);
                species.topFitness = file:read("*number");
                species.staleness = file:read("*number");
                local numGenomes = file:read("*number");
                for g=1,numGenomes do
                        local genome = newGenome();
                        table.insert(species.genomes, genome);
                        genome.fitness = file:read("*number");
                        genome.maxneuron = file:read("*number");
                        local line = file:read("*line");
                        while line ~= "done" do
                                genome.mutationRates[line] = file:read("*number");
                                line = file:read("*line");
                        end
                        local numGenes = file:read("*number");
                        for n=1,numGenes do
                                local gene = newLink();
                                table.insert(genome.links, gene);
                                local enabled;
                                gene.enter, gene.out, gene.weight, gene.id, enabled = file:read("*number", "*number", "*number", "*number", "*number");
                                if enabled == 0 then
                                        gene.enabled = false;
                                else
                                        gene.enabled = true;
                                end
                        end
                end
        end
        file:close();

        while fitnessAlreadyMeasured() do
                nextGenome();
        end
        initializeRun();
        pool.currentFrame = pool.currentFrame + 1;
end

function loadPool()
        local filename = forms.gettext(saveLoadFile);
        loadFile(filename);
end

function playTop()
        local maxfitness = 0;
        local maxs, maxg;
        for s,species in pairs(pool.species) do
                for g,genome in pairs(species.genomes) do
                        if genome.fitness > maxfitness then
                                maxfitness = genome.fitness;
                                maxs = s;
                                maxg = g;
                        end
                end
        end

        pool.currentSpecies = maxs;
        pool.currentGenome = maxg;
        pool.maxFitness = maxfitness;
        forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness));
        initializeRun();
        pool.currentFrame = pool.currentFrame + 1;
        return;
end

function onExit()
        forms.destroy(form);
end

writeFile("temp.pool");

event.onexit(onExit);

form = forms.newform(200, 260, "Fitness");
maxFitnessLabel = forms.label(form, "Max Fitness: " .. math.floor(pool.maxFitness), 5, 8);
showNetwork = forms.checkbox(form, "Show Map", 5, 30);
showMutationRates = forms.checkbox(form, "Show M-Rates", 5, 52);
restartButton = forms.button(form, "Restart", initializePool, 5, 77);
saveButton = forms.button(form, "Save", savePool, 5, 102);
loadButton = forms.button(form, "Load", loadPool, 80, 102);
saveLoadFile = forms.textbox(form, Filename .. ".pool", 170, 25, nil, 5, 148);
saveLoadLabel = forms.label(form, "Save/Load:", 5, 129);
playTopButton = forms.button(form, "Play Top", playTop, 5, 170);
hideBanner = forms.checkbox(form, "Hide Banner", 5, 190);


while true do
        local backgroundColor = 0xD0FFFFFF;
        if not forms.ischecked(hideBanner) then
                gui.drawBox(0, 0, 300, 26, backgroundColor, backgroundColor);
        end

        local species = pool.species[pool.currentSpecies];
        local genome = species.genomes[pool.currentGenome];

        if forms.ischecked(showNetwork) then
                displayGenome(genome);
        end

        if pool.currentFrame%5 == 0 then
                evaluateCurrent();
        end

        joypad.set(controller);

        getPositions();
        if marioX > rightmost then
                rightmost = marioX;
                timeout = TimeoutConstant;
        end

        timeout = timeout - 1;


        local timeoutBonus = pool.currentFrame / 4;
        if timeout + timeoutBonus <= 0 then
                local fitness = rightmost - pool.currentFrame / 2;
                if gameinfo.getromname() == "Super Mario World (USA)" and rightmost > 4816 then
                        fitness = fitness + 1000;
                end
                if gameinfo.getromname() == "Super Mario Bros." and rightmost > 3186 then
                        fitness = fitness + 1000;
                end
                if fitness == 0 then
                        fitness = -1;
                end
                genome.fitness = fitness;

                if fitness > pool.maxFitness then
                        pool.maxFitness = fitness;
                        forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness));
                        writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile));
                end

                console.writeline("Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " fitness: " .. fitness);
                pool.currentSpecies = 1;
                pool.currentGenome = 1;
                while fitnessAlreadyMeasured() do
                        nextGenome();
                end
                initializeRun();
        end

        local measured = 0;
        local total = 0;
        for _,species in pairs(pool.species) do
                for _,genome in pairs(species.genomes) do
                        total = total + 1;
                        if genome.fitness ~= 0 then
                                measured = measured + 1;
                        end
                end
        end
        if not forms.ischecked(hideBanner) then
                gui.drawText(0, 0, "Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " (" .. math.floor(measured/total*100) .. "%)", 0xFF000000, 11);
                gui.drawText(0, 12, "Fitness: " .. math.floor(rightmost - (pool.currentFrame) / 2 - (timeout + timeoutBonus)*2/3), 0xFF000000, 11);
                gui.drawText(100, 12, "Max Fitness: " .. math.floor(pool.maxFitness), 0xFF000000, 11);
        end

        pool.currentFrame = pool.currentFrame + 1;

        emu.frameadvance();
end
