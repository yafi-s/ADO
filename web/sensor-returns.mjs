// The legacy front radar plot stores XY and Doppler. Corner radars store XYZ and Doppler.
export function cornerReturns(frame) {
  return Object.values(frame?.corner_radar ?? {}).flatMap(rows => Array.isArray(rows)
    ? rows.filter(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite)) : []);
}
export function lidarCount(frame) {
  const points = frame?.lidar_xyz ?? frame?.lidar;
  return Array.isArray(points) ? points.length : null;
}
