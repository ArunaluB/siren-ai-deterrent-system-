# Wildlife Monitoring & Prediction System Guide

## Overview

The Wildlife Monitoring & Prediction System is an advanced platform designed to support wildlife officers in decision-making through:

- **Real-time Elephant Detection**: Continuous monitoring and detection of elephant activity
- **Unit Health Monitoring**: Track the health and status of all monitoring units
- **Future Predictions**: AI-powered predictions of elephant activity patterns
- **Data Collection & Learning**: Historical data collection that improves predictions over time
- **Decision Support**: Intelligent recommendations and alerts for wildlife officers
- **Advanced Analytics**: Comprehensive insights and pattern analysis

## Accessing the System

### Main Deterrence System
- **URL**: `http://localhost:5001/`
- **Purpose**: Real-time deterrence decision making and action selection

### Monitoring & Prediction Dashboard
- **URL**: `http://localhost:5001/monitoring`
- **Purpose**: Comprehensive monitoring, predictions, and decision support

## Features

### 1. System Overview Dashboard
- **Total Detections**: Cumulative count of all elephant detections
- **Active Units**: Number of operational monitoring units
- **System Health**: Overall system operational status
- **Active Alerts**: Current alerts requiring attention

### 2. Unit Health Monitoring
Each monitoring unit displays:
- **Battery Level**: Current battery status with visual indicator
- **Sensor Health**: Sensor system operational status
- **Detections (24h)**: Number of detections in last 24 hours
- **Uptime Percentage**: Unit reliability metric
- **Status**: Operational, Maintenance, or Offline

**Color Coding**:
- 🟢 Green: Healthy (battery > 50%, sensor > 80%)
- 🟡 Yellow: Warning (battery 30-50%, sensor 70-80%)
- 🔴 Red: Critical (battery < 30%, sensor < 70%)

### 3. Future Activity Predictions
- **24/48/72 Hour Forecasts**: Predict elephant activity patterns
- **Visual Charts**: Interactive charts showing predicted activity
- **Summary Statistics**: Average, peak, and minimum predicted counts
- **Confidence Levels**: Model confidence indicators

**How to Use**:
1. Select prediction timeframe (24/48/72 hours)
2. Click "Generate 24h Predictions"
3. View predictions on chart and summary statistics

### 4. Activity Trends
- **Historical Trends**: Visual representation of elephant activity over time
- **Daily Patterns**: Activity patterns by day
- **Trend Analysis**: Identify increasing or decreasing activity

### 5. Hourly Activity Patterns
- **24-Hour Distribution**: See when elephants are most active
- **Peak Hours Identification**: Automatically identifies peak activity times
- **Strategic Planning**: Use patterns to optimize patrol schedules

### 6. Decision Support Recommendations
The system provides intelligent recommendations based on:
- **Activity Trends**: Unusual increases in elephant activity
- **Unit Health**: Units requiring maintenance or attention
- **Habituation Risk**: Deterrent effectiveness degradation
- **Weather Conditions**: Environmental factors affecting operations
- **Pattern Analysis**: Strategic insights from historical data

**Priority Levels**:
- 🔴 **High**: Immediate action required (critical unit failures, significant activity spikes)
- 🟡 **Medium**: Attention needed (habituation risk, weather alerts)
- 🟢 **Low**: Informational (pattern insights, optimization suggestions)

### 7. Active Alerts
Real-time alerts for:
- **Low Battery**: Units with critically low battery
- **Sensor Degradation**: Units with sensor health issues
- **System Failures**: Operational problems requiring intervention

### 8. Comprehensive Analytics

#### Overview Tab
- Total detections
- Average elephant count
- Peak activity levels

#### Patterns Tab
- Hourly activity distribution
- Temporal patterns analysis
- Peak activity identification

#### Unit Performance Tab
- Individual unit statistics
- Detection rates per unit
- Uptime and reliability metrics
- Battery status tracking

#### Effectiveness Tab
- Deterrent effectiveness distribution
- Success/failure rates
- Action effectiveness analysis

## API Endpoints

### Monitoring Status
```
GET /api/monitoring/status
```
Returns overall system status including total detections, active units, and system health.

### Detections
```
GET /api/monitoring/detections?days=1
POST /api/monitoring/detections
```
- **GET**: Retrieve historical detections (optionally filtered by days)
- **POST**: Record a new elephant detection

### Unit Health
```
GET /api/monitoring/units
GET /api/monitoring/units/<unit_id>
PUT /api/monitoring/units/<unit_id>
```
- **GET /units**: Get health status of all units
- **GET /units/<id>**: Get specific unit health
- **PUT /units/<id>**: Update unit health status

### Predictions
```
GET /api/monitoring/predictions
POST /api/monitoring/predictions
```
- **GET**: Retrieve cached predictions
- **POST**: Generate new predictions (body: `{"n_hours": 24}`)

### Recommendations
```
GET /api/monitoring/recommendations
```
Returns decision support recommendations, alerts, and priority levels.

### Analytics
```
GET /api/monitoring/analytics
```
Returns comprehensive analytics including:
- Overview statistics
- Temporal patterns
- Unit performance metrics
- Effectiveness distributions

## Data Storage

The system stores data in the `data/` directory:
- `historical_data.json`: All detection records
- `unit_health.json`: Unit health status and metrics
- `predictions.json`: Cached prediction results

Data is automatically persisted and loaded on system startup.

## Auto-Refresh

The dashboard automatically refreshes every 30 seconds to show:
- Updated system status
- Latest unit health
- Recent detections
- New recommendations

## Best Practices

### For Wildlife Officers

1. **Daily Monitoring**:
   - Check system overview for overall status
   - Review active alerts and recommendations
   - Monitor unit health for maintenance needs

2. **Strategic Planning**:
   - Use hourly patterns to optimize patrol schedules
   - Review predictions for upcoming activity
   - Plan maintenance based on unit health

3. **Decision Making**:
   - Prioritize high-priority recommendations
   - Address critical alerts immediately
   - Use pattern insights for long-term planning

4. **Data Collection**:
   - Ensure all detections are recorded
   - Update unit health status regularly
   - Review analytics for trends

### For System Administrators

1. **Maintenance**:
   - Monitor unit battery levels
   - Schedule maintenance for degraded sensors
   - Review system health metrics

2. **Data Management**:
   - Historical data is automatically managed (last 1000 entries)
   - Backup data directory regularly
   - Monitor disk space usage

3. **Performance**:
   - System auto-refreshes every 30 seconds
   - Predictions improve with more historical data
   - Model retrains automatically as data accumulates

## Troubleshooting

### Dashboard Not Loading
- Check that the server is running on port 5001
- Verify all dependencies are installed
- Check browser console for errors

### No Predictions Available
- Ensure at least 10 detections are recorded
- Generate predictions using the "Generate 24h Predictions" button
- Check that historical data exists

### Units Not Showing
- Verify unit health data is initialized
- Check `data/unit_health.json` exists
- Restart the server to reinitialize units

### Recommendations Not Appearing
- Ensure historical data exists
- Check that unit health data is available
- Verify system has sufficient data for analysis

## Integration with Deterrence System

The monitoring system integrates with the main deterrence system:
- Detections from deterrence predictions are automatically recorded
- Historical data improves prediction accuracy
- Unit health affects deterrence recommendations
- Habituation tracking is shared between systems

## Future Enhancements

The system is designed to learn and improve over time:
- **Machine Learning**: Predictions improve with more data
- **Pattern Recognition**: Identifies complex activity patterns
- **Adaptive Recommendations**: Recommendations become more accurate
- **Predictive Maintenance**: Predicts unit failures before they occur

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review system logs
3. Verify data files in `data/` directory
4. Check API endpoint responses

---

**System Version**: 2.0  
**Last Updated**: 2024  
**Status**: Production Ready
