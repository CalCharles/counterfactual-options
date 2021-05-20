import numpy as np

EPSILON = 2

class ContingentObject():
    def __init__(self, dim):
        self.limits = np.array([0,0,0,0])
        self.vel = np.array([0] * dim)
        self.dim=dim
        self.isAction = False
        self.isSphere = False
        self.isAABB = False
        self.isRect = False
        self.isGripper = False  
        self.name = ""
        self.attribute = 1.0 # TODO: attribute should probably be array
        self.center = np.array([0] * dim)
        self.moved = False
        self.interaction_trace = list()
        self.sides = np.array([0] * (dim)) # length and width of the object

    def getContact(self, other):
        pass

    def actContact(self, vector, other):
        pass

    def move(self):
        pass

    def getPos(self, mid):
        return [int(mid[0] - (self.sides[0] / 2)), int(mid[1]  - (self.sides[1]/2))]

    def getMidpoint(self):
        return self.center

    def getVel(self):
        return self.vel

    def getAttribute(self):
        return self.attribute


class PhysicalObject(ContingentObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.vel = np.array([0] * dim)
        self.angular = np.array([0] * dim)
        self.bb  = np.array([0] * (dim * 2))
        self.sides = np.array([0] * (dim)) # length and width of the object
        self.pos = np.array([0]*dim)
        self.dim = dim
        self.radius = 0

    def updateBounding(self, center):
        self.pos = center
        self.bb = [center[i] - self.sides[i] / 2.0 for i in range(self.dim)] + [center[i] + self.sides[i] / 2.0 for i in range(self.dim)]

    def updateCenter(self):
        for i in range(self.dim):
            self.pos[i] = (self.bb[i] + self.bb[i + self.dim])/2.0

    def getMidpoint(self):
        self.updateCenter()
        return self.pos

    def getVel(self):
        return self.vel

    def getContact(self, other): # only supporting AABBs and spheres
        if self.isAction:
            return other.applyAction(self)
        if self.isAABB and other.isAABB:
            return self.AABBcontact(other)
        elif self.isSphere and other.isAABB:
            return self.SphereAABBcontact(other)
        elif other.isSphere and self.isAABB:
            return other.SphereAABBcontact(self)
        else:
            return None

    def AABBcontact(self, other):
        '''
        gets the contact between two bounding boxes (2d only)
        '''
        sxl, syl, sxh, syh = self.bb
        oxl, oyl, oxh, oyh = other.bb
        ax = (sxl <= oxh and sxh >= oxl)
        ay = (syl <= oyh and syh >= oyl)
        if ax and ay:
            return self.pos - other.pos
        else:
            return None

    def SphereAABBcontact(self, other):
        '''
        gets the contact between two bounding boxes (2d only)
        '''
        sx, sy = self.pos
        sr = self.radius
        oxl, oyl, oxh, oyh = other.bb
        x = max(oxl, min(sx, oxh))
        y = max(oyl, min(sy, oyh))
        d = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        # print(d)
        if d < sr:
            return self.pos - other.pos
        else:
            return None


    def setmove(self, vel, angular):
        self.vel = vel
        self.angular = angular

    def move(self):
        if np.sum(np.abs(self.vel)) > 0:
            self.moved = True
            self.bb[:self.dim] += self.vel
            self.bb[-self.dim:] += self.vel
            self.updateCenter()
            # self.vel = np.array([0] * self.dim)
        else:
            self.moved = False
        # TODO: angular velocity

    def actContact(self, vector, other):
        '''
        changes features based on contact
        '''
        return None

    def applyAction(self, actions):
        return None

    def pushed(self, other):
        vec = other.pos - self.pos
        # mag = np.dot(vec / np.linalg.norm(vec +.001), self.vel)
        # print(vec, mag)
        if np.dot(vec / np.linalg.norm(vec +.001), self.vel) > 0:
            other.setmove(self.vel, 0)
            other.interaction_trace.append(self.name)

class Action(ContingentObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.name = "Action"
        self.attribute = 0

    def updateBounding(self, pos):
        self.pos = [0,0]
        self.bb = [0] * 4


class Gripper(PhysicalObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.gripped = None
        self.isGripper = True
        self.name = "Gripper"
        self.fixed = False

    def actContact(self, contact, other):
        if contact is not None: # we are in contact, but possibly not the cente
            if self.gripped is not None or self.fixed: # already grabbing
                if self.gripped is not other: # in contact with a different object
                    self.pushed(other)
                else:
                    other.setmove(self.vel, self.angular)
            else:
                # TODO: not hacked
                d = [0,0]
                d[0] = contact[0] - (other.sides[0]/2 - 1)
                d[1] = contact[1]
                # print(d, np.linalg.norm(d))
                if np.linalg.norm(d) <= EPSILON:
                    self.gripped = other
                    other.gripped = True

class CartesianGripper(Gripper):
    def __init__(self, dim):
        super().__init__(dim)
        self.limits = [50,6,78,78]
        self.sides = [8,8]
        self.isAABB = True

    def applyAction(self, action_obj):
        self.setmove(np.array([0,0]), None)
        if action_obj.attribute == 1:
            if self.pos[0] > self.limits[0]:
                self.setmove(np.array([-1,0]), None)
        if action_obj.attribute == 2:
            if self.pos[0] < self.limits[self.dim]:
                self.setmove(np.array([1,0]), None)
        if action_obj.attribute == 3:
            if self.pos[1] > self.limits[1]:
                self.setmove(np.array([0,-1]), None)
        if action_obj.attribute == 4:
            if self.pos[1] < self.limits[1 + self.dim]:
                self.setmove(np.array([0,1]), None)

class CartesianPusher(Gripper):
    def __init__(self, dim):
        super().__init__(dim)
        # self.limits = [50,6,78,78]
        self.limits = [6,6,78,78]
        self.sides = [8,8]
        self.opening = [8-np.round(8 * .3),8-np.round(8 * .3)]
        self.isAABB = True
        self.fixed = True

    def applyAction(self, action_obj):
        self.setmove(np.array([0,0]), None)
        self.interaction_trace.append(action_obj.name)
        if action_obj.attribute == 1:
            if self.pos[0] > self.limits[0]:
                self.setmove(np.array([-1,0]), None)
        if action_obj.attribute == 2:
            if self.pos[0] < self.limits[self.dim]:
                self.setmove(np.array([1,0]), None)
        if action_obj.attribute == 3:
            if self.pos[1] > self.limits[1]:
                self.setmove(np.array([0,-1]), None)
        if action_obj.attribute == 4:
            if self.pos[1] < self.limits[1 + self.dim]:
                self.setmove(np.array([0,1]), None)
        # print(action_obj.attribute, self.vel)
    
    def pushed(self, other):
        vec = other.pos - self.pos
        total_vel = [0,0]
        # print(vec, other.center, self.center, self.vel, (other.sides + self.sides)/2)
        for i in range(2):
            if np.abs(self.vel[i]) > 0:
                # if to the correct side and not sliding past
                o = (i+1)%2
                if vec[i] * self.vel[i] > 0 and np.abs(other.pos[o] - self.pos[o]) < (other.sides[o] + self.sides[o])/2:
                    total_vel[i] = self.vel[i]
        # TODO: within gripper no-pushing
        if type(other) != Target:
            other.interaction_trace.append(self.name)
            other.setmove(np.array(total_vel), 0)
        else:
            total_vel = self.vel.tolist()
            for i in range(2):
                if np.abs(self.vel[i]) > 0:
                    # if to the correct side and not sliding past
                    o = (i+1)%2
                    if vec[i] * self.vel[i] > 0 and np.abs(other.pos[o] - self.pos[o]) < (other.sides[o] + self.sides[o])/2:
                        total_vel[i] = 0
            self.setmove(np.array(total_vel), None)
        # mag = np.dot(vec / np.linalg.norm(vec +.001), self.vel)
        # print(vec, mag)


# class Joint(Gripper):


# class Arm(PhysicalObject):
#     def __init__(self, **kwargs):
#         super().__init__(self, kwargs['dim'])
#         self.joints = [Joint()] * (2 * kwargs['num_joints'])
    
#     def applyAction(self, action):
#         self.setmove(np.array([0,0]), None)
#         if action == 1:
#             if self.center[0] < self.limits[0]:
#                 self.setmove(np.array([1,0]), None)
#         if action == 2:
#             if self.center[0] > self.limits[self.dim]:
#                 self.setmove(np.array([-1,0]), None)
#         if action == 3:
#             if self.center[1] < self.limits[1]:
#                 self.setmove(np.array([0,1]), None)
#         if action == 4:
#             if self.center[1] > self.limits[1 + self.dim]:
#                 self.setmove(np.array([0,-1]), None)

class Stick(PhysicalObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.gripped = None
        self.limits = np.array([35,6,50,78])
        self.sides = np.array([50,4])
        self.isAABB = True
        self.isSphere = False
        self.name = "Stick"

    def actContact(self, contact, other):
        if contact is not None and type(other) != Target: # we are in contact, but possibly not the center
            self.pushed(other)

class Target(PhysicalObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.limits = np.array([3,5,9,78])
        # self.sides = np.array([6,6])
        self.sides = np.array([10,10])
        self.gripped = None
        self.isAABB = True
        self.isSphere = False
        self.touched = False
        self.name = "Target"

    def actContact(self, contact, other):
        # print(contact)
        if contact is not None: # we are in contact, but possibly not the center
            if type(other) == Block or type(other) == Sphere:
                self.touched = True
                self.attribute = 2
                self.interaction_trace.append(other.name)


class Sphere(PhysicalObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.isSphere = True
        self.limits = np.array([7,19,30,72])
        self.radius = 4
        self.sides = np.array([4,4])
        self.name = "Sphere"

class Block(PhysicalObject):
    def __init__(self, dim):
        super().__init__(dim)
        self.isAABB = True
        self.limits = np.array([16,16,72,64])
        # self.limits = np.array([7,24,30,68])
        # self.sides = np.array([4,4])
        self.sides = np.array([6,6])
        self.name = "Block"