# Drone Racer Implementation Todos

## Current Scene Configuration Analysis ✅
- Read and analyzed Isaac Sim scene configuration
- Identified 4-gate setup in `drone_racer_env_cfg.py:39-46`
- Analyzed gate generation system in `track_generator.py`
- Reviewed command generation system in `commands.py`

## Single Gate Modification Requirements

### Implementation Options

#### Option 1: Single Gate with Dynamic Angles
- Modify track_config in `drone_racer_env_cfg.py` to use single gate
- Update `_resample_command` in `commands.py` for dynamic angle generation
- Expand pose_range for different approach angles

#### Option 2: Virtual Intermediate Waypoints (Recommended)
- Add `_get_intermediate_target` method to `GateTargetingCommand` class
- Generate virtual waypoints without physical gates
- Maintain current task structure while providing angle variety

#### Option 3: Dynamic Track Generator
- Create new function `generate_single_track_with_dynamic_angles`
- Support dynamic rotation capability
- Keep base gate orientation static, update angles dynamically

## Key Implementation Points

### Code Locations to Modify
- `drone_racer_env_cfg.py:39-46` - Track configuration
- `commands.py:135-160` - Command resampling
- `track_generator.py` - Optional: Add dynamic generation function

### Preservation Requirements
- Keep current reward system intact
- Maintain gate passing logic (`commands.py:177-200`)
- Preserve existing training framework
- Keep collision detection system

### Technical Details
- Gate passing detection uses plane projection method
- Reward system includes coordinated flight, gate passing bonuses
- FPV recording capability available but currently disabled
- Randomizable starting positions already supported

## Next Steps
1. Choose implementation option (Virtual waypoints recommended)
2. Implement single gate configuration
3. Add dynamic angle generation logic
4. Test with existing training pipeline
5. Verify reward system compatibility

## Notes
- Current setup uses kinematic gates with gravity disabled
- Gate size parameter: 1.5 meters (configurable)
- Starting position ranges: x(-3.5,-1.5), y(-0.5,0.5), z(1.5,0.5)
- Episode length: 20 seconds
- Simulation dt: 1/400 seconds