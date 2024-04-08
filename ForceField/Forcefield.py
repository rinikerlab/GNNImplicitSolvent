from openmm.app import ForceField
from openmm.app import PME, NoCutoff, GBn2
from openmm.unit import nanometer, elementary_charge, angstrom
from openmm.app import HBonds
from openmm.openmm import CustomGBForce, GBSAOBCForce, CustomBondForce, CustomNonbondedForce, CustomExternalForce
import numpy as np
from copy import deepcopy

from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from openmm.app.internal.customgbforces import *
from openmm.app.internal.customgbforces import _createEnergyTerms

class _generic_force_field:

    def __init__(self, force_field : ForceField = None):
        self._openmm_forcefield = force_field
        self._ready_for_usage = False

    @property
    def name(self):
        return str(self)

    @property
    def scale_needed(self):
        return False

    @property
    def water_model(self):
        return "n.a."

    @property
    def ready(self):
        return self._ready_for_usage

    @ready.setter
    def ready(self,ready):
        self._ready_for_usage = ready

    @property
    def forcefield(self):
        return self._openmm_forcefield

    def create_system(self, topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer):

        if self.water_model == "implicit":
            return self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        else:
            print("built explicit System",nonbondedMethod,nonbondedCutoff)
            return self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)


    def __str__(self):
        return self.__class__.__name__


class OpenFF_forcefield(_generic_force_field):


    def __init__(self,pdb_id,solvent_model='TIP3P',cache=None):
        
        solute_smiles = pdb_id.split('_in_')[0]
        solvent_smiles = pdb_id.split('_in_')[1]
        self._solvent_model = solvent_model.lower()

        self._solute = Molecule.from_smiles(solute_smiles,allow_undefined_stereo=True)
        if solvent_smiles != 'v':
            self._solvent = Molecule.from_smiles(solvent_smiles)
        if (solvent_smiles == 'O'):
            smirnoff = SMIRNOFFTemplateGenerator(molecules=[self._solute],forcefield='openff-2.0.0',cache=cache)
            forcefield = ForceField('%s.xml' % self._solvent_model)
            print('Water considered %s' % self._solvent_model)
        elif (solvent_smiles == 'v'):
            smirnoff = SMIRNOFFTemplateGenerator(molecules=[self._solute],forcefield='openff-2.0.0',cache=cache)
            forcefield = ForceField()
        else:
            smirnoff = SMIRNOFFTemplateGenerator(molecules=[self._solvent,self._solute],forcefield='openff-2.0.0',cache=cache)
            forcefield = ForceField()

        forcefield.registerTemplateGenerator(smirnoff.generator)

        topology = Topology()
        topology.add_molecule(self._solute)
        topology = topology.to_openmm()
        
        for res in topology.residues():
            smirnoff.generator(forcefield,res)
    
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer):
        return super().create_system(topology, nonbondedMethod, nonbondedCutoff)

    def __str__(self):
        return "openff200_" + self._solvent_model

    @property
    def water_model(self):
        return "explicit"

class OpenFF_forcefield_vacuum(OpenFF_forcefield):

    def create_system(self, topology, nonbondedMethod=NoCutoff,nonbondedCutoff=1 * nanometer):
        return super().create_system(topology, NoCutoff)

    def __str__(self):
        return "openff200_vacuum"

    @property
    def water_model(self):
        return "implicit"



class OpenFF_forcefield_vacuum_plus_custom(OpenFF_forcefield):

    def __init__(self, pdb_id,custom_force,force_name='GNN',cache=None):
        super().__init__(pdb_id,cache=cache)
        self._custom_force = custom_force
        self._force_name = force_name

    def create_system(self, topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer):
        system = super().create_system(topology, NoCutoff)
        custom_force = deepcopy(self._custom_force)
        system.addForce(custom_force)
        return system

    def __str__(self):
        return "openff200_vacuum_plus_" + self._force_name

    @property
    def water_model(self):
        return "implicit"


class OpenFF_forcefield_GBNeck2(OpenFF_forcefield):

    def __init__(self, pdb_id, solvent_model='TIP3P',SA=None,cache=None):
        super().__init__(pdb_id, solvent_model,cache=cache)
        self._SA = SA

    def create_system(self, topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer):

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        charges = np.array([system.getForces()[0].getParticleParameters(i)[0]._value for i in range(topology._numAtoms)])

        force = GBSAGBn2Force(cutoff=None,SA=self._SA,soluteDielectric=1)
        gbn2_parameters = np.empty((topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = charges # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters

        self._Data = gbn2_parameters
        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system.addForce(force)

        return system

    def __str__(self):
        if self._SA is None:
            return "openff200_GBNeck2"
        else:
            return "openff200_SAGBNeck2"

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class OpenFF_forcefield_SAGBNeck2(OpenFF_forcefield_GBNeck2):

    def __init__(self, pdb_id, solvent_model='TIP3P', SA='ACE', cache=None):
        super().__init__(pdb_id, solvent_model, SA, cache)


class OpenFF_forcefield_GBNeck2_e4(OpenFF_forcefield):

    def __init__(self, pdb_id, solvent_model='TIP3P',SA=None):
        super().__init__(pdb_id, solvent_model)
        self._SA =SA

    def create_system(self, topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer):

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        charges = np.array([system.getForces()[0].getParticleParameters(i)[0]._value for i in range(topology._numAtoms)])

        force = GBSAGBn2Force(cutoff=None,SA=self._SA,soluteDielectric=1,solventDielectric=4)
        gbn2_parameters = np.empty((topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = charges # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters

        self._Data = gbn2_parameters
        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system.addForce(force)

        return system

    def __str__(self):
        return "openff200_GBNeck2_epsilon_4"

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data


class Vacuum_force_field_plus_custom(_generic_force_field):

    def __init__(self,custom_force,force_name='GNN'):
        super().__init__(force_field = ForceField('amber99sbildn.xml'))
        self._ready_for_usage = True
        self._custom_force = custom_force
        self._force_name = force_name

    def __str__(self):
        return "vacuum_plus_" + self._force_name

    def create_system(self, topology, nonbondedMethod=None,
        nonbondedCutoff=1*nanometer):

        custom_force = deepcopy(self._custom_force)
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        system.addForce(custom_force)
        return system

    @property
    def water_model(self):
        return "implicit"

class Vacuum_force_field_plus_custom_plus_dummy_GB(_generic_force_field):

    def __init__(self,custom_force,data):
        super().__init__(force_field = ForceField('amber99sbildn.xml'))
        self._ready_for_usage = True
        self._custom_force = custom_force
        self._Data = data

    def __str__(self):
        return "custom_plus_dummyGB"

    def create_system(self, topology, nonbondedMethod=None,
        nonbondedCutoff=1*nanometer):

        custom_force = deepcopy(self._custom_force)
        custom_force.setForceGroup(2)
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        system.addForce(custom_force)

        force = GBSAGBn2Force(cutoff=None, SA=None, soluteDielectric=1,solventDielectric=1.0001/1)

        gbn2_parameters = np.empty((topology.getNumAtoms(), 6))
        gbn2_parameters[:, 0] = self._Data[:, 0]  # Charges
        gbn2_parameters[:, 1:] = force.getStandardParameters(topology)  # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        force.setForceGroup(3)
        system.addForce(force)

        return system

    @property
    def water_model(self):
        return "implicit"

class Vacuum_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field = ForceField('amber99sbildn.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "vacuum"

    @property
    def water_model(self):
        return "implicit"

class GBSAOBC_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field=ForceField('amber99sbildn.xml', 'amber99_obc.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "GBSAOBC"

    @property
    def water_model(self):
        return "implicit"

class CHARMM_GB_force_field(_generic_force_field):
    def __init__(self):
        super().__init__(force_field=ForceField('charmm36.xml', '//localhome/kpaul/Downloads/openmm-master/wrappers/python/openmm/app/data/implicit/obc2.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "CHARMM_OBC"

    @property
    def water_model(self):
        return "implicit"

class TIP5P_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field=ForceField('amber99sbildn.xml', 'tip5p.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "TIP5P"

    @property
    def water_model(self):
        return "tip5p"

class TIP4P_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field=ForceField('amber99sbildn.xml', 'tip4p.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "TIP4P"

    @property
    def water_model(self):
        return "tip4p"

class TIP3P_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field=ForceField('amber99sbildn.xml', 'tip3p.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "TIP3P"

    @property
    def water_model(self):
        return "tip3p"

class TIP3P_99SB_force_field(_generic_force_field):

    def __init__(self):
        super().__init__(force_field=ForceField('amber99sb.xml', 'tip3p.xml'))
        self._ready_for_usage = True

    def __str__(self):
        return "TIP3P_99SB"

    @property
    def water_model(self):
        return "tip3p"

class GB_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "-0.5*138.935456*(1/1-1/78.3)*q^2/B_ACE"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.3)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B_ACE1*B_ACE2*exp(-r^2/(4*B_ACE1*B_ACE2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or; or=radius-0.009"

        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)

        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]) * elementary_charge, np.double(self._Data[i, 1]) * nanometer,
                 np.double(self._Data[i, 2])]

            force.addParticle(input)


        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_OBC"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class test_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "-0.5*138.935456*(1/1-1/78.3)*q^2/B_ACE"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.3)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B_ACE1*B_ACE2*exp(-r^2/(4*B_ACE1*B_ACE2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2 + aaa/1 ;"

        text_gen = ["a","b","c","d","e","f","g","h","i","j"]

        texts = []
        for i in text_gen:
            for j in text_gen:
                for k in text_gen:
                    texts.append(i+j+k)

        values = np.random.rand(len(text_gen)**5 - 1)

        for i in range(len(text_gen)**5 - 1):
            I_value_expression += texts[0] +"*= " + str(values[i]) + ";"
            # I_value_expression += texts[i] +"="+ texts[i+1] + "* "+ str(values[i]) +"+ 1;"
        I_value_expression += texts[0] + " = or1;"

        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or; or=radius-0.009"

        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)

        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]) * elementary_charge, np.double(self._Data[i, 1]) * nanometer,
                 np.double(self._Data[i, 2])]

            force.addParticle(input)


        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "test"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_HCT_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):
        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create HCT Force
        force = GBSAHCTForce(cutoff=cutoff,SA=None)
        gbn2_parameters = np.empty((topology.getNumAtoms(),3))
        gbn2_parameters[:,0] = self._Data[:,0] # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_HCT"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_HCT_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):
        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create HCT Force
        force = GBSAHCTForce(cutoff=cutoff,SA='ACE')
        gbn2_parameters = np.empty((topology.getNumAtoms(),3))
        gbn2_parameters[:,0] = self._Data[:,0] # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_HCT"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_Neck2_force_field(_generic_force_field):
    #equivalent to amber igb=8

    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create GBn2 Force
        #force = GBSAGBn2Force(cutoff=cutoff,SA=None,soluteDielectric=1)
        force = GBSAGBn2Force(cutoff=None,SA=None,soluteDielectric=1)

        gbn2_parameters = np.empty((topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = self._Data[:,0] # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters
        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_Neck2"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_Neck2_force_field_plus_ML(_generic_force_field):
    def __init__(self,Data=None,torchforce=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        self._torchforce = torchforce
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create GBn2 Force
        #force = GBSAGBn2Force(cutoff=cutoff,SA=None,soluteDielectric=1)
        force = GBSAGBn2Force(cutoff=None,SA=None,soluteDielectric=1)

        gbn2_parameters = np.empty((topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = self._Data[:,0] # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(topology) # GBNeck2 parameters
        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)
        custom_force = deepcopy(self._torchforce)
        system.addForce(custom_force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_Neck2_plus_ML"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data


class GBSA_Neck2_force_field(_generic_force_field):
    def __init__(self, Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field=forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
                      nonbondedCutoff=1 * nanometer):
        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create GBn2 Force
        force = GBSAGBn2Force(cutoff=cutoff, SA='ACE')
        gbn2_parameters = np.empty((topology.getNumAtoms(), 6))
        gbn2_parameters[:, 0] = strip_unit(self._Data[:, 0]*elementary_charge,elementary_charge)  # Charges
        gbn2_parameters[:, 1:] = force.getStandardParameters(topology)  # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff,
                                                      constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_Neck2"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_Neck_force_field(_generic_force_field):
    def __init__(self, Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field=forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
                      nonbondedCutoff=1 * nanometer):
        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create GBn2 Force
        force = GBSAGBnForce(cutoff=cutoff, SA='ACE')
        gbn2_parameters = np.empty((topology.getNumAtoms(), 3))
        gbn2_parameters[:, 0] = self._Data[:, 0]  # Charges
        gbn2_parameters[:, 1:] = force.getStandardParameters(topology)  # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff,
                                                      constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_Neck"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_Neck_force_field(_generic_force_field):
    def __init__(self, Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field=forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
                      nonbondedCutoff=1 * nanometer):
        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create GBn2 Force
        force = GBSAGBnForce(cutoff=cutoff, SA=None)
        gbn2_parameters = np.empty((topology.getNumAtoms(), 3))
        gbn2_parameters[:, 0] = self._Data[:, 0]  # Charges
        gbn2_parameters[:, 1:] = force.getStandardParameters(topology)  # GBNeck2 parameters

        # Add Particles and finalize force
        force.addParticles(gbn2_parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff,
                                                      constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_Neck"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_OBC_int_force_field(_generic_force_field):
    def __init__(self, Data=None,version=1,SA=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        self._SA = SA
        self._version = version
        super().__init__(force_field=forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
                      nonbondedCutoff=1 * nanometer):

        cutoff = strip_unit(nonbondedCutoff, angstrom)
        # Create OBC Force
        if self._version == 1:
            force = GBSAOBC1Force(cutoff=cutoff, SA=self._SA)
        elif self._version == 2:
            force = GBSAOBC2Force(cutoff=cutoff, SA=self._SA)
        else:
            exit('Only Version 1 or 2 valid')
        parameters = np.empty((topology.getNumAtoms(), 3))
        parameters[:, 0] = self._Data[:, 0]  # Charges
        parameters[:, 1:] = force.getStandardParameters(topology)  # GBOBC parameters
        #parameters[:, 1] = self._Data[:, 2]*0.1
        #parameters[:, 2] = self._Data[:, 3]
        #parameters[:, 3] = 1 #self._Data[:, 1]
        # Add Particles and finalize force
        force.addParticles(parameters)
        force.finalize()

        # Create System and add force
        system = self._openmm_forcefield.createSystem(topology=topology, nonbondedMethod=NoCutoff,
                                                      constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GB_OBC_int"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_OBC1_force_field(GBSA_OBC_int_force_field):
    def __init__(self,Data):
        super().__init__(Data=Data,version=1,SA=None)
    def __str__(self):
        return "GB_OBC1"

class GBSA_OBC1_force_field(GBSA_OBC_int_force_field):
    def __init__(self,Data):
        super().__init__(Data=Data,version=1,SA='ACE')
    def __str__(self):
        return "GBSA_OBC1"

class GB_OBC2_force_field(GBSA_OBC_int_force_field):
    def __init__(self,Data):
        super().__init__(Data=Data,version=2,SA=None)
    def __str__(self):
        return "GB_OBC2"

class GBSA_OBC2_force_field(GBSA_OBC_int_force_field):
    def __init__(self,Data):
        super().__init__(Data=Data,version=2,SA='ACE')
    def __str__(self):
        return "GBSA_OBC2"

class GBSA_OBC_ACE_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # custom->addGlobalParameter("solventDielectric", obc->getSolventDielectric());
        # custom->addGlobalParameter("soluteDielectric", obc->getSoluteDielectric());

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B_ACE)^6-0.5*138.935456*(1/1-1/78.3)*q^2/B_ACE"
        #single_energy_expression = "2.25936*(radius+0.14)^2*(radius/B_ACE)^6-0*0.5*138.935456*(1/1-1/78.45)*q^2/B_ACE"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.3)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B_ACE1*B_ACE2*exp(-r^2/(4*B_ACE1*B_ACE2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or; or=radius-0.009"

        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)



        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]) * elementary_charge, np.double(self._Data[i, 1]) * nanometer,
                 np.double(self._Data[i, 2])]

            force.addParticle(input)


        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=NoCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_OBC_ACE"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_ACE_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("B")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B_ACE)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or; or=radius-0.009"

        # Calculate parameters
        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        # Add custom energy function
        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1),np.double(self._Data[i, 2]*0.1),np.double(self._Data[i, 3])]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_ACE"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_ACE_I_scaling_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("I_scaling")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I*or*I_scaling; or=radius-0.009"

        # Calculate parameters
        force.addComputedValue("I", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B", B_value_expression, force.SingleParticle)

        # Add custom energy function
        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0])*elementary_charge, np.double(self._Data[i, 1]),np.double(self._Data[i, 2]*0.1) * nanometer,np.double(self._Data[i, 3])]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_ACE_I_scale"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_ACE_I_scaling_2_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("charge")
        force.addPerParticleParameter("I_scaling")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "-0.5*138.935485*(1/1-1/78.45)*charge^2/B"

        # Pairwise Interactions
        pair_energy_expression = "-138.935485*(1/1-1/78.45)*charge1*charge2/f;f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I*or; or=radius-0.009;"

        # Calculate parameters
        #force.addComputedValue("I", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        #force.addComputedValue("B", B_value_expression, force.SingleParticle)

        force.addComputedValue("I",
                               "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                               "U=r+sr2;"
                               "L=max(or1, D);"
                               "D=abs(r-sr2);"
                               "sr2 = scale2*or2;"
                               "or1 = radius1-0.009; or2 = radius2-0.009;", CustomGBForce.ParticlePairNoExclusions)

        force.addComputedValue("B", "1/(1/or-tanh(0.8*psi+2.909125*psi^3)/radius);"
                                   "psi=I*or; or=radius-0.009;", CustomGBForce.SingleParticle)

        # Add custom energy function
        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0])*elementary_charge, np.double(self._Data[i, 1]),np.double(self._Data[i, 2]*0.1) * nanometer,np.double(self._Data[i, 3])]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_ACE_I_scale"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_ACE_I_scaling_force_field_no_SASA(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("I_scaling")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "-0.5*138.935456*(1/1-1/78.45)*q^2/B_ACE"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B_ACE1*B_ACE2*exp(-r^2/(4*B_ACE1*B_ACE2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        #                    "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or*I_scaling; or=radius-0.009"

        # Calculate parameters
        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        # Add custom energy function
        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0])*elementary_charge, np.double(self._Data[i, 1]),np.double(self._Data[i, 2]*0.1) * nanometer,np.double(self._Data[i, 3])]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_ACE_I_scale_no_SASA"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_ACE_born_scaling_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("B_scaling")
        force.addPerParticleParameter("radius")
        force.addPerParticleParameter("scale")

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B_ACE)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B_ACE"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B_ACE1*B_ACE2*exp(-r^2/(4*B_ACE1*B_ACE2)));"

        # Use ACE approach to estimate
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "B_scaling * 1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I_ACE*or; or=radius-0.009"

        # Calculate parameters
        force.addComputedValue("I_ACE", I_value_expression, force.ParticlePairNoExclusions) # No exclusions for Born radius calc
        force.addComputedValue("B_ACE", B_value_expression, force.SingleParticle)

        # Add custom energy function
        force.addEnergyTerm(pair_energy_expression,force.ParticlePair)   # Do check for exclusions here
        force.addEnergyTerm(single_energy_expression,force.SingleParticle)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]), np.double(self._Data[i, 1]),np.double(self._Data[i, 2]*0.1),np.double(self._Data[i, 3])]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA_ACE_born_scale"

    @property
    def scale_needed(self):
        return True

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_one_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("B")
        force.addPerParticleParameter("radius")

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B"

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"

        # One NEEDS to calculate values otherwise program fails not used
        force.addComputedValue("q12","q1*q2",1)
        force.addComputedValue("qB", "q*B",0)

        force.addEnergyTerm(single_energy_expression,type=0)
        force.addEnergyTerm(pair_energy_expression,type=1)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1),np.double(self._Data[i, 2]*0.1)]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA"

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSAOBC_custom_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # get Force
        force = CustomGBForce()
        force.addPerParticleParameter("q")
        force.addPerParticleParameter("scale")
        force.addPerParticleParameter("radius")

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B"
        # SA contribution from Schaefer and coworkers with modification from Jay Ponder

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"

        # One NEEDS to calculate values otherwise program fails not used
        I_value_expression = "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
        I_value_expression += "U=r+sr2;"
        I_value_expression += "C=2*(1/or1-1/L)*step(sr2-r-or1);"
        I_value_expression += "L=max(or1, D);"
        I_value_expression += "D=abs(r-sr2);"
        I_value_expression += "sr2 = scale2*or2;"
        I_value_expression += "or1 = radius1-0.009; or2 = radius2-0.009"

        B_value_expression = "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
        B_value_expression += "psi=I*or; or=radius-0.009"

        # add value 1 = Pairlis, 0 = Single
        force.addComputedValue("I", I_value_expression, 1)
        force.addComputedValue("B", B_value_expression, 0)

        force.addEnergyTerm(single_energy_expression,type=0)
        force.addEnergyTerm(pair_energy_expression,type=1)


        for i in range(len(self._Data)):
            input = [np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1),np.double(self._Data[i, 2]*0.1)]
            force.addParticle(input)

        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)
        system.addForce(force)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA"

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GBSA_force_field(_generic_force_field):
    def __init__(self,Data=None):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # Single Interactions
        single_energy_expression = "28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/1-1/78.45)*q^2/B"
        sf = CustomExternalForce(single_energy_expression)
        sf.addPerParticleParameter("q")
        sf.addPerParticleParameter("B")
        sf.addPerParticleParameter("radius")

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"
        pf = CustomNonbondedForce(pair_energy_expression)
        pf.addPerParticleParameter("q")
        pf.addPerParticleParameter("B")

        for i in range(len(self._Data)):
            pf.addParticle((np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1)))
            index = sf.addParticle(i,(np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1),np.double(self._Data[i, 2]*0.1)))
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)

        #system.get_forces()[0].

        system.addForce(sf)
        system.addForce(pf)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "GBSA"

    @property
    def water_model(self):
        return "implicit"

    @property
    def Data(self):
        return self._Data

    @Data.setter
    def Data(self,Data):
        self._Data = Data

class GB_2_force_field(_generic_force_field):
    def __init__(self,Data):
        forcefield = ForceField('amber99sbildn.xml')
        self._Data = Data
        super().__init__(force_field = forcefield)
        self._ready_for_usage = True

    def create_system(self, topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer):

        # Single Interactions
        single_energy_expression = "138.93545764438207*-0.5*(1/1-1/78.45)*q^2/B;"
        sf = CustomExternalForce(single_energy_expression)
        sf.addPerParticleParameter("q")
        sf.addPerParticleParameter("B")

        # Pairwise Interactions
        pair_energy_expression = "-138.93545764438207*(1/1-1/78.45)*q1*q2/f;" # unit cor + kJ/mol const =(1.602176634*10**-19)**2/(8.8541878128*4*np.pi*10**-21)*10**23*6.02214076/1000
        pair_energy_expression += "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));"
        pf = CustomNonbondedForce(pair_energy_expression)
        pf.addPerParticleParameter("q")
        pf.addPerParticleParameter("B")
        pf.setCutoffDistance(nonbondedCutoff)

        for i in range(len(self._Data)):
            pf.addParticle((np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1)))
            index = sf.addParticle(i,(np.double(self._Data[i, 0]), np.double(self._Data[i, 1]*0.1)))
        system = self._openmm_forcefield.createSystem(topology=topology,nonbondedMethod=nonbondedMethod,
                                                    nonbondedCutoff=nonbondedCutoff,constraints=HBonds)

        system.addForce(sf)
        system.addForce(pf)

        return system

    def adapt_GB_values(self,system,Data):
        pass

    def __str__(self):
        return "customGB"

    @property
    def water_model(self):
        return "implicit"

class GB_Neck2_scale_force_field(GBSAGBn2Force):

    def __init__(self):
        super().__init__(solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None, kappa=0.0)

    def calculate_radii(self):
        gbn2_parameters = np.empty((topology.getNumAtoms(), 6))
        gbn2_parameters[:, 0] = self._Data[:, 0]  # Charges
        gbn2_parameters[:, 1:] = force.getStandardParameters(topology)  # GBNeck2 parameters

    def _addEnergyTerms(self):
        self.addPerParticleParameter("charge")
        self.addPerParticleParameter("or")  # Offset radius
        self.addPerParticleParameter("sr")  # Scaled offset radius
        self.addPerParticleParameter("alpha")
        self.addPerParticleParameter("beta")
        self.addPerParticleParameter("gamma")
        self.addPerParticleParameter("radindex")

        n = len(self._uniqueRadii)
        m0Table = self._createUniqueTable(m0)
        d0Table = self._createUniqueTable(d0)
        self.addTabulatedFunction("getd0", Discrete2DFunction(n, n, d0Table))
        self.addTabulatedFunction("getm0", Discrete2DFunction(n, n, m0Table))

        self.addComputedValue("I", "Ivdw+neckScale*Ineck;"
                                   "Ineck=step(radius1+radius2+neckCut-r)*getm0(radindex1,radindex2)/(1+100*(r-getd0(radindex1,radindex2))^2+"
                                   "0.3*1000000*(r-getd0(radindex1,radindex2))^6);"
                                   "Ivdw=select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                                   "U=r+sr2;"
                                   "L=max(or1, D);"
                                   "D=abs(r-sr2);"
                                   "radius1=or1+offset; radius2=or2+offset;"
                                   "neckScale=0.826836; neckCut=0.68; offset=0.0195141",
                              CustomGBForce.ParticlePairNoExclusions)

        self.addComputedValue("B", "1/(1/or-tanh(alpha*psi-beta*psi^2+gamma*psi^3)/radius);"
                                   "psi=I*or; radius=or+offset; offset=0.0195141", CustomGBForce.SingleParticle)
        _createEnergyTerms(self, self.solventDielectric, self.soluteDielectric, self.SA, self.cutoff, self.kappa,
                           self.OFFSET)

