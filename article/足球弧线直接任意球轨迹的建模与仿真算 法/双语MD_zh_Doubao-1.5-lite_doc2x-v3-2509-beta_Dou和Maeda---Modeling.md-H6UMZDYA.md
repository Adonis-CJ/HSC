双语MD\_zh\_Doubao-1.5-lite\_doc2x-v3-2509-beta\_Dou和Maeda - Modeling.md

# Modeling and Simulating Algorithms for Track of Direct Free-kick in Football Arc

# 足球弧线直接任意球轨迹的建模与仿真算法

Yueqi Dou, Masato Maeda

窦月琪，前田正人

School of Human Development and Environment , KOBE University ,Japan

日本神户大学人类发展与环境学院

ABSTRACT. The description of the sports state of soccer in sports is complicated. We analyze the force on the soccer's motion process from two angles of plane and curved surface respectively, and describe its motion track, especially the causes of soccer's arc motion, so as to explain the motion state of balls such as soccer. Football curve ball is also called "banana ball". In football matches, it can make it difficult for the defending side to make a correct judgment. Therefore, it has a fascinating overall effect in the competition and is a very popular technology in contemporary football technology. On the basis of literature review, the factors such as ejection angle, rotation and air resistance of arc ball are analyzed, and the stress process of football in flight is decomposed.A scientific model was established without considering other secondary factors, and the "hot zone" that can shoot the arc ball was analyzed, and the best shot area of the arc ball was given. At the same time, the football trajectory is simulated. The flight path of the curved ball is reproduced by input parameters to objectively determine whether the soccer ball hits the goal.

摘要：体育运动中足球运动状态的描述较为复杂。我们分别从平面和曲面两个角度分析足球运动过程中的受力情况，并描述其运动轨迹，特别是足球弧线运动的成因，从而解释足球等球类的运动状态。足球弧线球也被称为“香蕉球”。在足球比赛中，它会使防守方难以做出正确判断。因此，它在比赛中具有迷人的整体效果，是当代足球技术中一项非常受欢迎的技术。在文献综述的基础上，分析了弧线球的射出角度、旋转和空气阻力等因素，分解了足球飞行过程中的受力过程。在不考虑其他次要因素的情况下建立了科学模型，分析了能射出弧线球的“热点区域”，并给出了弧线球的最佳射门区域。同时，对足球轨迹进行了模拟。通过输入参数再现弧线球的飞行路径，客观判断足球是否能命中球门。

KEYWORDS: Football Curve; Direct Free-kick; Motion Trajectory; Modeling

关键词：足球弧线；直接任意球；运动轨迹；建模

## 1. Introduction

## 1. 引言

Football is the world's most popular sport with millions of participants. Because it has received such extensive attention, many people have been studying the technologies involved\[1]. Of the 171 goals in the 1998 World Cup, 42 were set goals, of which ${50}\%$ were direct free kicks,thus showing the role of an accurate free kick in football. Beckham is good at banana ball, while Cristiano Ronaldo is good at falling the ball in front of the goal or landing the rebound ball\[2]. All these make us cannot help studying the wonderful phenomenon of curling in the football world. When we are playing football, the soccer ball, after receiving the force that does not pass through the center of gravity, then rotates and flies in a curve in the air\[3]. This is called arc ball, also known as "banana ball"\[4]. The left deviation of the ball from the normal orbit is called "left-hand" and vice versa. Curved balls are usually kicked out on the medial instep and the lateral instep\[5].

足球是世界上最受欢迎的运动，有数以百万计的参与者。由于受到如此广泛的关注，许多人一直在研究其中涉及的技术\[1]。在1998年世界杯的171个进球中，有42个是定位球，其中${50}\%$个是直接任意球，从而显示了精确任意球在足球中的作用。贝克汉姆擅长香蕉球，而克里斯蒂亚诺·罗纳尔多则擅长在球门前下坠或接住反弹球\[2]。所有这些都让我们不禁去研究足球世界中精彩的弧线现象。当我们踢足球时，足球在受到不通过重心的力后，会旋转并在空中呈曲线飞行\[3]。这被称为弧线球，也被称为“香蕉球”\[4]。球从正常轨道向左偏离称为“左旋”，反之亦然。弧线球通常用脚内侧和脚外侧踢出\[5]。

In recent years, many experts and scholars have developed a strong interest in this technology\[6]. In a ball sports project, when the ball moves in the air, it is often accompanied by rotation, and the rotation of the ball itself necessarily affects the trajectory of its movement. In other words, in this case, the Magnus effect needs to be considered\[7]. The arc ball in the football game has a significant effect on the Magnus effect. In the current game, the arc ball is generally used to improve the hit rate\[8]. Therefore, the direct free kick in the arc will be the research object, and the better kicking point will be analyzed. By studying the arc ball in football, the force of the rotating sphere in the air is analyzed. Simplify the football, model the football trajectory according to the parameters, analyze the better area of the direct free kick kicking out the arc, and discuss with the actual situation\[9]. At the same time, the football trajectory is changed by constantly changing the parameters. Judging whether to hit the goal, providing a theoretical basis for the football player to kick a high-quality, high-scoring direct free kick\[10].

近年来，许多专家学者对这项技术产生了浓厚兴趣\[6]。在一项球类运动项目中，当球在空中运动时，通常会伴随着旋转，而球本身的旋转必然会影响其运动轨迹。换句话说，在这种情况下，需要考虑马格努斯效应\[7]。足球比赛中的弧线球对马格努斯效应有显著影响。在当前比赛中，弧线球通常用于提高命中率\[8]。因此，将弧线内的直接任意球作为研究对象，分析更好的击球点。通过研究足球中的弧线球，分析旋转球体在空中的受力情况。简化足球，根据参数对足球轨迹进行建模，分析踢出弧线的直接任意球的更佳区域，并与实际情况进行讨论\[9]。同时，通过不断改变参数来改变足球轨迹。判断是否能命中球门，为足球运动员踢出高质量、高得分的直接任意球提供理论依据\[10]。

### 2.The Theoretical Basis of the Curve Movement of Football

### 2. 足球曲线运动的理论基础

### 2.1 Mechanics Principle of Curving Ball in Football

### 2.1 足球中弧线球的力学原理

According to the law of rotation, when an object rotates around an axis, the vector sum of its torque to the axis must not be zero. The force of kicking a ball does not pass through the center of the ball, that is, the eccentric force. Because the resultant force can be decomposed into normal force, making the ball translate. Tangential force turns the ball. The fast and slow rotation of the ball depends on the size and direction of the kicking force. When the force is constant, the faster the ball rotates, but the reverse force decreases correspondingly, and the horizontal speed of the ball also decreases. This situation is not conducive to long-distance passing in the match.When the direction of the resultant line of action is constant, the greater the force, the faster the ball's rotation speed and flight speed. Therefore, the power of playing is very important. The standard ball moment of inertia for the game is constant, so the greater the moment, the greater the angular acceleration. From the above analysis, we can see that the force line does not kick the football through the center of the center of the ball, it is with a strong rotation. This kind of ball with a rotating sphere flying in the air, rotating at a high speed, makes it close to the air particles of the ball skin, and rotates to form a circulation layer, which rotates with the ball.

根据旋转定律，当物体绕轴旋转时，其对轴的力矩矢量和一定不为零。踢球的力不通过球心，即偏心力。因为合力可分解为法向力，使球平移。切向力使球转动。球旋转的快慢取决于踢球力的大小和方向。当力恒定时，球旋转得越快，但反向力相应减小，球的水平速度也减小。这种情况不利于比赛中的长传。当合力作用线方向恒定时，力越大，球的旋转速度和飞行速度越快。所以踢球的力量很重要。比赛用的标准球转动惯量是恒定的，所以力矩越大，角加速度越大。从上述分析可知，力线不通过球心踢球，球会强烈旋转。这种旋转的球体在空中飞行，高速旋转，使其靠近球皮的空气颗粒，旋转形成环流层，随球一起转动。

### 2.2 Magnus effect

### 2.2 马格努斯效应

Most of the current college physics textbooks take particles as the research object when discussing the oblique projectile motion, and seldom involve the oblique projectile motion of rotating objects. In fact, the motion of rotating projectile is a very common phenomenon. Especially in ball games, when the ball is in air movement, it often rotates with itself. For example, soccer, table tennis, basketball, tennis, etc. However, the rotation of the sphere itself will inevitably affect the trajectory of its motion, that is to say, in this case, Magnus effect needs to be considered.When a cylinder rotates about its own axis and fluid flows in a direction perpendicular to the axis, it is subjected to a lateral force perpendicular to the flow direction. The direction of the force is always directed to the same side from the side opposite to the linear velocity on the cylindrical surface. This phenomenon is called the Magnus effect. In fact, the Magnus effect is not limited to a rotating cylinder. The so-called "banana ball" in football matches, the arc ball in the table tennis match and the ball-cutting technique all have a significant effect on the Magnus effect.

当前大多数大学物理教材在讨论斜抛运动时，都以质点为研究对象，很少涉及旋转物体的斜抛运动。实际上，旋转抛体的运动是很常见的现象。尤其是在球类运动中，球在空中运动时，常常会自身旋转。比如足球、乒乓球、篮球、网球等。然而，球体自身的旋转不可避免地会影响其运动轨迹，也就是说，在这种情况下，需要考虑马格努斯效应。当圆柱体绕自身轴线旋转且流体沿垂直于轴线的方向流动时，它会受到一个垂直于流动方向的侧向力。力的方向总是从圆柱面上与线速度相反的一侧指向同一侧。这种现象称为马格努斯效应。实际上，马格努斯效应并不局限于旋转的圆柱体。足球比赛中的所谓“香蕉球”、乒乓球比赛中的弧圈球以及削球技术都对马格努斯效应有显著影响。

## 3. Factors Influencing Football Goals

## 3. 影响足球射门的因素

### 3.1 Projection angle factor

### 3.1 投射角度因素

There are many factors that affect the flight path of football. Such as kick technique, rotation, firing angle, kick position, etc. There are also wind speed, air temperature, humidity and other weather factors. In addition, the physical characteristics of the football, such as roughness, quality, number of pieces of skin that make up the surface of the ball, material and inflation pressure, will also affect the flight trajectory of the football. Under ideal conditions, the model is established without considering the influence of external factors, but these factors sometimes play a vital role in the actual kick training. Therefore, these factors should be taken into account.For example, an object of mass m has a greater influence on the maximum range throw angle in the range of less than $3\mathrm{\;{kg}}$ ,and the effect is less pronounced after more than $3\mathrm{\;{kg}}$ . As the mass $\mathrm{m}$ increases,the projection angle is finally ${45}^{ \circ }$ . In addition,under the same conditions,the upper spiral has a low arc,a fast drop, and a near drop point. The lower spin is flying at a high arc and the drop point is far, and no rotating ball is in between. This is due to the presence or absence of key factors in the model with different lift and lift directions.

影响足球飞行轨迹的因素有很多。比如踢球技术、旋转、发射角度、踢球位置等。还有风速、气温、湿度等天气因素。此外，足球的物理特性，如粗糙度、质量、构成球表面的皮块数量、材料和充气压力等，也会影响足球的飞行轨迹。在理想条件下，建立模型时不考虑外部因素的影响，但这些因素在实际踢球训练中有时会起到至关重要的作用。因此，应该考虑这些因素。例如，质量为 m 的物体在小于$3\mathrm{\;{kg}}$的范围内对最大射程投射角度影响较大，超过$3\mathrm{\;{kg}}$后影响不太明显。随着质量$\mathrm{m}$增加，投射角度最终为${45}^{ \circ }$。此外，在相同条件下，上旋球弧线低、下落快、落点近。下旋球飞行弧线高、落点远，不旋转的球介于两者之间。这是由于模型中不同升力和升力方向的关键因素的有无导致的。

### 3.2 Air resistance factor

### 3.2 空气阻力因素

The magnitude of air resistance is related to the velocity of the object relative to the air and the shape of the object. In general, the air resistance of the flying body is proportional to the air density, the surface area of the flying body and the waiting speed, and inversely proportional to the smoothness of the surface of the flying body. The streamline shape of the flying body also affects the resistance. The better the streamline shape, the smaller the resistance. The pressure difference caused by the rotation of "banana ball" is called lift. It is related to the motion speed and rotation frequency of the sphere and is expressed by the following equation (1).The ball speed is a variable that is full of variables and difficult to determine. The first thing to note is that since the lift is perpendicular to the ball speed, the presence of lift has no effect on the magnitude of the ball speed. Secondly, because of the existence of the ball speed, the air resistance occurs, and the air resistance in turn causes the ball speed to change. The flight path of the ball is deviated from the arc shape, and the actual trajectory depends on how the speed changes. Air resistance is a dynamic quantity that is related to the shape of the object, the speed of motion, and the surface characteristics.

空气阻力的大小与物体相对于空气的速度以及物体的形状有关。一般来说，飞行物体的空气阻力与空气密度、飞行物体的表面积和等待速度成正比，与飞行物体表面的光滑程度成反比。飞行物体的流线型形状也会影响阻力。流线型形状越好，阻力越小。“香蕉球”旋转所引起的压力差称为升力。它与球体的运动速度和旋转频率有关，并由以下方程（1）表示。球速是一个充满变量且难以确定的变量。首先需要注意的是，由于升力垂直于球速，升力的存在对球速的大小没有影响。其次，由于球速的存在，会产生空气阻力，而空气阻力又会导致球速发生变化。球的飞行轨迹偏离弧形，实际轨迹取决于速度如何变化。空气阻力是一个动态量，它与物体的形状、运动速度和表面特性有关。

${\mathrm{R}}_{\mathrm{i}} = {\mathrm{M}}_{\mathrm{i}}\mathop{\sum }\limits_{{\mathrm{i} = 1}}^{{\mathrm{N}}_{\mathrm{r}}}{\widehat{\mathrm{R}}}_{\mathrm{i},\mathrm{r}} \qquad \text{(1)}$

Where Ri is the lift coefficient, $\mathrm{{Nr}}$ is the air density, $\mathrm{r}$ is the diameter of the ball, I is the ball rotation frequency, and Mi is the ball speed. It is Ri that makes the ball move in an arc.

其中Ri是升力系数，$\mathrm{{Nr}}$是空气密度，$\mathrm{r}$是球的直径，I是球的旋转频率，Mi是球速。是Ri使球作弧线运动。

### 4.Arc Motion of Football in Plane and Surface

### 4. 足球在平面和曲面中的弧线运动

### 4.1 The Curve Movement of Football in Plane

### 4.1 足球在平面中的曲线运动

The wonderful football match ended slowly and many classic pictures still appeared in front of the fans. One of the classics is the beautiful arc when a star shoots. Are you eager to try? As long as your hitting leg accelerates forward, the longus hallucis muscle fastens to the ankle. Keep your feet straight. Pay attention to your toes when hitting the ball. The thigh naturally swings to the vertical and the calf swings to a large extent. Make full use of waist and abdomen strength to hit the lower side of the ball with the instep. The line of action of the force deviates from the center of the ball when hitting the ball. When the football flies from your feet, it will draw a beautiful arc in the air. Why do you play curling when playing football? In the past, people usually started from the ideal plane motion to analyze the force on the soccer ball. Here, they further carried out qualitative force analysis from two aspects of plane and curved surface to describe its motion trajectory.

精彩的足球比赛缓缓落幕，许多经典画面仍浮现在球迷眼前。其中一个经典就是球星射门时那美妙的弧线。你是否渴望一试？只要你的击球腿加速前摆，拇长肌固定于脚踝。保持双脚伸直。击球时注意你的脚趾。大腿自然摆动至垂直，小腿大幅摆动。充分利用腰腹力量，用脚背击打足球的下侧。击球时力的作用线偏离球心。当足球从你脚下飞起时，它会在空中划出一道美丽的弧线。为什么踢足球时会踢出弧线球呢？过去，人们通常从理想的平面运动开始分析足球上的力。在此，他们进一步从平面和曲面两个方面进行定性的受力分析，以描述其运动轨迹。

Force F plays soccer. The line of force passes through the center of the soccer ball. The soccer ball flies forward at speed V. Due to the viscosity of the air, the air near the football surface is driven to flow. Air has a backward speed relative to football, which is different from the forward speed and the backward speed of flying football. The front velocity is smaller than the rear velocity, causing the front air pressure and the rear air pressure to be different. Their pressure difference acts on the spherical surface of the soccer ball, making the soccer ball subject to pressure drag. Ignoring the fact that the effect of the gas height difference before and after the football is not significant, according to Bernoulli equation (2), the pressure difference acts on the effective area of the ball surface of the football, and its direction is along the opposite direction of the flying direction of the football, thus hindering the flying football. The size of the pressure drag is related to the difference in air velocity between the front and rear surfaces of the flying soccer ball. The difference in air flow rate changes the pressure difference, thus changing the size of the pressure drag. From the above analysis, it can be seen that football is affected by pressure drag and its own gravity during flight.

力F作用于足球。力的作用线穿过足球的中心。足球以速度V向前飞行。由于空气的粘性，足球表面附近的空气被带动流动。空气相对于足球有一个向后的速度，这与飞行足球的向前速度和向后速度不同。前方速度小于后方速度，导致前后空气压力不同。它们的压力差作用于足球的球面，使足球受到压力阻力。忽略足球前后气体高度差的影响不显著这一事实，根据伯努利方程（2），压力差作用于足球球面的有效面积上，其方向沿足球飞行方向的相反方向，从而阻碍飞行的足球。压力阻力的大小与飞行足球前后表面空气速度的差异有关。空气流速的差异改变压力差，从而改变压力阻力的大小。从上述分析可以看出，足球在飞行过程中受到压力阻力和自身重力的影响。

$V = \sqrt{\frac{F \times {Vt}}{K} + \frac{{F}^{2}}{4 \times {K}^{2}}} + {Vt} + \frac{F}{2 \times K} \qquad \text{(2)}$

The flying football generates a vertical acceleration of gravity under the action of gravity, and produces a tangential acceleration in the horizontal direction under the action of the differential pressure resistance. The combination of the two accelerations causes the motion of the football to be a curve motion in the plane K, so the football does not turn into an arc motion within the curved surface.

飞行的足球在重力作用下产生垂直向下的重力加速度，在压差阻力作用下在水平方向产生切向加速度。这两个加速度的合成使得足球在平面K内的运动为曲线运动，所以足球在曲面内不会变成弧线运动。

### 4.2 Curve Motion of Football in Surface

### 4.2 足球在曲面中的曲线运动

If the line of action of the force deviates from the center of the ball, the force can be translated to the center of the ball. At this time, the football is equivalent to the combined action of the force passing through the center of the ball and the force generating a moment to make the football rotate. When this force acts on the lower left side of the soccer ball, the soccer ball will fly in the air at an initial speed along a direction at an angle with the plane clip. At the same time, the ball rotates counterclockwise at a certain angular velocity. If this force acts on the lower right side of the football, the football will rotate clockwise. The rotation of the soccer ball and the viscous effect of the air cause the circulation of the air on the surface of the soccer ball, and the circulation rate is consistent with the rotation direction of the soccer ball.When the ball is flying forward at a relative speed, the air flow flows backward at a certain speed with respect to the soccer ball. According to the Bernoulli equation, according to the following equation (3), if the effective area of the pressure difference acting on the soccer sphere is S, then the magnitude of the force the football receives is F, that is, the football is subjected to the lateral force F. Lift. As mentioned above, the airflow velocity component is different in the front and rear speeds of the flying soccer ball, so that the soccer ball is subjected to the pressure difference resistance.

如果力的作用线偏离球心，该力可平移至球心。此时，足球相当于受到通过球心的力和产生使足球旋转的力矩的力的共同作用。当此力作用于足球的左下方时，足球将以初始速度沿与平面剪辑成一定角度的方向在空中飞行。同时，球以一定角速度逆时针旋转。如果此力作用于足球的右下方，足球将顺时针旋转。足球的旋转和空气的粘性作用导致足球表面空气的环流，环流速率与足球的旋转方向一致。当球以相对速度向前飞行时，气流相对于足球以一定速度向后流动。根据伯努利方程，根据以下方程（3），如果作用在足球球体上的压力差的有效面积为S，那么足球所受的力的大小为F，即足球受到侧向力F。升力。如前所述，气流速度分量在飞行足球的前后速度中不同，从而使足球受到压力差阻力。

${\mathrm{R}}_{\mathrm{j}} = \frac{{\mathrm{f}}_{\mathrm{{ij}}}}{\mathrm{{Tj}}} \times {\mathrm{S}}_{\mathrm{j}} \qquad \text{(3)}$

It can be seen from this that the flying football will be affected by the combined force of lift, pressure drag and gravity.

由此可见，飞行中的足球将受到升力、压力阻力和重力的合力影响。

5.  Optimal Modeling and Simulation of Track of Direct Free-kick in Football Arc
6.  足球弧线直接任意球轨迹的优化建模与仿真

### 5.1 Optimization of Basic Model for Track of Arc Directly Arbitrary Sphere

### 5.1 弧线直接任意球轨迹基本模型的优化

During the translational flight of a soccer ball, it is only affected by gravity and air resistance, which causes the ball speed to decrease. Curved direct free kick flies in rotation, not only under the action of gravity and air resistance, but also under the differential pressure perpendicular to the flight direction. The initial trajectory of soccer is slightly curved and gradually increases. Ge Longqi analyzed the forces on football and established a basic model (4). The right-angle coordinate axis is established vertically upward, and the axis is along the original advancing direction. The axis represents the lateral offset.Let the ball of mass $\mathrm{m}$ be kicked out at a certain initial speed and rotated around the axis passing through the center of the ball at $\mathrm{x}$ as the initial rotational angular velocity, and the basic model of the motion law of the ball is obtained. In addition, the modeling optimizes the football direct free-ball trajectory base model (4). The optimization model scheme (5) adds the displacement in the i-axis direction and the influence factors of the horizontal initial velocity and the x-direction declination. The situation discussed in the base model is limited to horizontal displacement, which is clearly not realistic.

在足球的平移飞行过程中，它仅受重力和空气阻力影响，这会导致球速下降。弧线直接任意球在旋转中飞行，不仅受到重力和空气阻力作用，还受到垂直于飞行方向的压差作用。足球的初始轨迹略有弯曲且逐渐增大。葛龙麒分析了足球受力情况并建立了基本模型（4）。垂直向上建立直角坐标轴，轴沿原来的前进方向。轴表示横向偏移。设质量为$\mathrm{m}$的球以一定初始速度踢出，并以$\mathrm{x}$为初始旋转角速度绕通过球心的轴旋转，得到球的运动规律基本模型。此外，建模对足球直接任意球轨迹基本模型（4）进行了优化。优化模型方案（5）增加了i轴方向的位移以及水平初始速度和x方向偏角的影响因素。基本模型中讨论的情况仅限于水平位移，这显然不符合实际。

${\mathrm{p}}_{\mathrm{{ij}}} = {\mathrm{x}}_{\mathrm{{ij}}}^{\prime }/\mathop{\sum }\limits_{{\mathrm{i} = 1}}^{\mathrm{m}}{\mathrm{x}}_{\mathrm{{ij}}}^{\prime } \qquad \text{(4)}$

$\mathrm{B}\left( \overrightarrow{\mathrm{X}}\right) = \mathop{\prod }\limits_{{\mathrm{i} = 1}}^{\mathrm{m}}{\left( {\mathrm{f}}_{\mathrm{i}}\left( \overrightarrow{\mathrm{X}}\right) - {\mathrm{f}}_{\mathrm{i}}\left( {\overrightarrow{\mathrm{X}}}_{\mathrm{w}}\right) \right) }^{1/\mathrm{m}} \qquad \text{(5)}$

Based on the above analysis, the vertical displacement needs to be added to the basic model, which is treated as a vertical upward throwing motion here.

基于上述分析，需要在基本模型中加入垂直位移，这里将其视为垂直上抛运动。

### 5.2 Simulation of football trajectory model

### 5.2 足球轨迹模型的仿真

For curved direct free kick, this paper first gives a more general description of the three-dimensional motion track of curved free kick, giving a more intuitive understanding of curved direct free kick. Firstly, a kick center is created, and the basic model is used to simulate the football in three directions. In order to know which part of the football field the arc direct free kick hits better, first divide the football field half into uniform grids and establish the origin of coordinates. Grid intersections are set as kick points and 4 parameters are set. Initial velocity, angular velocity and kick point. Since no free kick can be kicked in the small restricted area and the pitch is symmetrical, the kick point is only considered in a certain dark shaded area. According to the model established by the optimization scheme, through the constant change of the parameters in a certain range of values, we can get at which points the scoring rate of arc-line direct free kick is the highest, which is what we call the hot zone, and get its coordinates. Then import coordinate data and draw contour lines of the hot zone.

对于弧线直接任意球，本文首先对弧线任意球的三维运动轨迹给出更一般的描述，以便对弧线直接任意球有更直观的理解。首先创建一个击球中心，并使用基本模型在三个方向上对足球进行仿真。为了知道弧线直接任意球在足球场的哪部分命中效果更好，首先将足球场半场划分为均匀网格并建立坐标原点。将网格交点设为击球点并设置4个参数。初始速度、角速度和击球点。由于在小禁区内无法踢出任意球且球场是对称的，所以仅在某个深色阴影区域考虑击球点。根据优化方案建立的模型，通过在一定取值范围内不断改变参数，我们可以得到弧线直接任意球在哪几个点的得分率最高，即我们所说的热点区域，并得到其坐标。然后导入坐标数据并绘制热点区域的等高线。

In addition, three-dimensional simulation of the direct free-arbit trajectory of the arc is performed, and it is determined whether or not the goal is scored by inputting different parameters. The specific operation is as follows. Simulate a soccer field and a soccer ball, and realize the three-dimensional simulation by presenting different perspectives of the soccer field simulation interface by changing the parameters "azimuth" and "top view angle". Set the parameters of the kick point and the parameters of the ball trajectory. The parameters are the translation speed, the rotation speed, the angle between the translation speed and the axis, the angle between the horizontal speed and the axis, and the initial coordinates. By changing the parameters, the trajectory of the football is determined and whether or not the goal is scored. At the same time, add the tick options "Show Motion Track" and "Pre-Retention Track" to achieve more user-friendly operation options. Add model selection blocks, such as base models and other optimization scheme models. Three models can be arbitrarily selected for soccer trajectory simulation, and the comparison of football trajectories under different scheme models is realized. Set the "Simulation" button and click to run the football track.

此外，对弧线的直接任意球轨迹进行三维模拟，并通过输入不同参数来判断是否进球。具体操作如下。模拟一个足球场和一个足球，并通过改变“方位角”和“顶视图角度”参数来呈现足球场模拟界面的不同视角，从而实现三维模拟。设置击球点参数和球轨迹参数。这些参数包括平移速度、旋转速度、平移速度与轴之间的角度、水平速度与轴之间的角度以及初始坐标。通过改变参数，确定足球的轨迹以及是否进球。同时，添加勾选选项“显示运动轨迹”和“预保留轨迹”，以实现更用户友好的操作选项。添加模型选择块，如基础模型和其他优化方案模型。可以任意选择三个模型进行足球轨迹模拟，实现不同方案模型下足球轨迹的比较。设置“模拟”按钮并点击以运行足球轨迹。

### 5.3 Training of Curving Ball in Football

### 5.3 足球中弧线球的训练

In order to ensure the effectiveness of the curve ball in the competition, it is required that the curve ball kicked should not only have sufficient flying distance, but also have strong rotation. To this end, the following teaching methods should be adopted in learning and training. First of all, students (or athletes) should be given a theoretical explanation of the mechanics principle. In order to ensure that the kicked ball has a strong rotation, they should practice kicking the rotating ball on the basis of mastering the inside and outside of the instep. For example, when kicking a ball with the inside of the instep, there are three main points of action. Kicking feet should hit the back and outside of the ball, manic joints should be turned inward and feet should be slightly tilted upward. If you use the outside of the instep to kick the ball, pay attention to the toe turning inward, the toe fastening downward, and the instep stretching straight. The above two kicking methods must have the "rubbing ball" action to make the ball rotate. After practicing skillfully, we should exert more force to make the ball have enough forward initial speed and the swing range of the foot is larger, and the kick leg should have the feeling of "sending out" the ball along with the kick direction, so as to ensure the rotation speed of the football.

为了确保弧线球在比赛中的有效性，要求踢出的弧线球不仅要有足够的飞行距离，还要有强烈的旋转。为此，在学习和训练中应采用以下教学方法。首先，应向学生（或运动员）对力学原理进行理论讲解。为了确保踢出的球有强烈的旋转，他们应在掌握脚背内侧和外侧的基础上练习踢旋转球。例如，当用脚背内侧踢球时，有三个主要作用点。踢球脚应击打在球的后部和外侧，踝关节应向内转动，脚应稍微向上倾斜。如果用脚背外侧踢球，注意脚趾向内转动，脚趾向下扣紧，脚背伸直。上述两种踢球方法都必须有“摩擦球”的动作以使球旋转。熟练练习后，应加大力度以使球有足够的向前初始速度，且脚的摆动幅度更大，踢球腿应有将球“送出”与踢球方向一致的感觉，以确保足球的旋转速度。

By analyzing and establishing a reasonable and effective mathematical model, we use the contour map to show the better area to kick the arc direct free kick. At the same time, through the analysis of the curve ball, the soccer movement track is drawn and the simulation model of the soccer track is completed. By inputting various parameters and using the established model, the running track of this kind of curve ball is displayed to judge whether to score a goal. Although the arc direct free kick, the "hot spot" of the kick point and the three-dimensional plane of the direct free kick are simulated, there are many factors that affect the football flight path in real life, and there is still room for improvement. For example, it is possible to increase the analysis of the influencing factors of football, such as weather conditions, kicking skills, and the physical characteristics of football.

通过分析并建立合理有效的数学模型，我们用等高线图展示踢出弧线直接任意球的较好区域。同时，通过对弧线球的分析，绘制足球运动轨迹并完成足球轨迹的模拟模型。通过输入各种参数并使用建立的模型，显示这种弧线球的运行轨迹以判断是否进球。虽然对弧线直接任意球、击球点的“热点”以及直接任意球的三维平面进行了模拟，但在现实生活中影响足球飞行路径的因素很多，仍有改进空间。例如，可以增加对足球影响因素的分析，如天气状况、踢球技巧和足球的物理特性。

## 6. Conclusion

## 6. 结论

Flying soccer balls generate vertical gravitational acceleration under the action of gravity, horizontal tangential acceleration under the action of pressure drag, and horizontal normal acceleration under the action of lift. The combined acceleration of gravity acceleration and tangential acceleration makes the football move in a curve in the plane ${\mathrm{P}}^{\prime }$ ,while the reverse acceleration makes the football no longer move in a curve in the plane ${\mathrm{P}}^{\prime }$ ,which urges the football to turn left. Therefore,on the lower side of football, the line of action of force deviates from the center of the ball, and the kicked football will draw a beautiful arc in the air.In this paper, a reasonable and optimized mathematical model is established by analysis, and the contour map is used to show the better area of the direct free kick of the kick line. Through the analysis of the arc ball, the trajectory of the football is drawn, and the simulation model of the football trajectory is completed. Using the established model, the trajectory of such a curved ball is described to determine whether or not to score. According to the obtained better area of the goal and the modeling and simulation of the input parameters, the purpose of improving the goal of the athlete is achieved.

飞行中的足球在重力作用下产生垂直重力加速度，在压力阻力作用下产生水平切向加速度，在升力作用下产生水平法向加速度。重力加速度和切向加速度的合成加速度使足球在平面${\mathrm{P}}^{\prime }$内做曲线运动，而反向加速度使足球在平面${\mathrm{P}}^{\prime }$内不再做曲线运动，促使足球向左转弯。因此，在足球的下侧，力的作用线偏离球心，踢出的足球将在空中画出一条优美的弧线。本文通过分析建立了合理优化的数学模型，并用等高线图展示了罚球线直接任意球的较好区域。通过对弧线球的分析，绘制了足球的轨迹，完成了足球轨迹的模拟模型。利用建立的模型描述这种弧线球的轨迹以确定是否进球。根据得到的较好进球区域以及对输入参数的建模和模拟，实现了提高运动员进球率的目的。

## References

\[1] Slota,Adam(2014).Bezier Curve Based Programmed Trajectory for Coordinated Motion of Two Robots in Cartesian Space. Applied Mechanics and Materials, no.555, pp.192-198.

\[2] Beh.J, Han.D, Ko.H(2014).Rule-based trajectory segmentation for modeling hand motion trajectory. Pattern Recognition, vol.47,no.4,pp.1586-1601.

\[3] Shao.Z, Li.Y(2016).On Integral Invariants for Effective 3-D Motion Trajectory Matching and Recognition.IEEE Transactions on Cybernetics,vol.46, no.2, pp.511-523.

\[4] Choi.Y,Kim.D,Hwang.S, et al(2017).Dual-arm robot motion planning for collision avoidance using B-spline curve. International Journal of Precision Engineering and Manufacturing, vol.18, no.6, pp.835-843.

\[5] Z.Shao,Y.Li(2015).Integral invariants for space motion trajectory matching and recognition. Pattern Recognition,vol.48, no. 8, pp.2418-2432.

\[6] Y. Zhang, H. Xu, W. Ru(2015).A deployment trajectory design method based on the Bezier curves. Journal of Computational and Theoretical Nanoscience, vol. 12, pp. 5288-5296.

\[7] C.J. Li, C.L.Liu, G.C.Wang, et al(2014).A Fast Trajectory Planning Algorithm Research for Point-to-Point Motion. Advanced Materials Research, no.940, pp.526-530.

\[8] J.Yuan, W\.Yao, P.Zhao,et al(2015).Kinematics and trajectory of both-sides cylindrical lapping process in planetary motion type. International Journal of Machine Tools and Manufacture, no.92, pp.60-71.

\[9] X.Yang,M. Li(2017).Study on the curvature of the particle motion trajectory in ultra-precision lapping and polishing.Optical Technique,vol.43, no.4, pp.289- 293.

\[10] Z.Sun,Z.Wang,S.J.Phee(2015).Modeling and motion compensation of a bidirectional tendon-sheath actuated system for robotic endoscopic surgery. Computer Methods and Programs in Biomedicine, vol.119, no.2, pp.77-87.
