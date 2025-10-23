1. automatic video recording.
2. automatic naming by reward and scene.
   1. that means scene needs to be named.
3. is there any scripts that support scene changing tests？
4. is it possible to training inherited from some weights?
5. is this `demo` entrypoint automatically using best_agent.pt?
6. could we now inherited from best trajectory tracker?
7. if not model-based, the model tends to be memorize its behavoir according to the scene.
8. is issac sim can save the scene editing to config file?



您的课程学习 (CL) 方案 (如您之前提到的)：

阶段1 (最易)：将无人机模型简化为质点 (质点)，在无碰撞的环境中学习通过目标点。

阶段2 (中等)：引入更真实的动力学模型和逐步增大的碰撞体积 (碰撞体积逐步增大)，学习在简单赛道中避障。

阶段3 (最难)：使用最终的、大尺寸的飞机模型 (大尺寸飞机)，在复杂的赛道中竞速。