; Creates the IDL reference tables for irispy/utils/tests/test_bursts.py (see overview.txt).
; Needs SolarSoft with $SSW_IRIS set and this path (read_iris_l2 calls create_struct_temp from vobs/ontology):
;   !path = expand_path('+~/ssw/iris/idl') + ':' + expand_path('+~/ssw/gen/idl') + ':' $
;     + expand_path('+~/ssw/vobs/ontology/idl') + ':' + !path
; Run it in a directory holding
; iris_l2_20130902_182935_4000005156_raster_t000_r00000.fits, with the unzipped
; iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits in
; $IRIS_DATA/level2/2013/09/02/20130902_163935_4000255147/, as iris_sji_burst_check finds it by date.

; The shortest string that reads back as the same double, which is how astropy writes floats.
function shortest, value
  compile_opt idl2
  for n = 1, 17 do begin
    s = string(double(value), format='(F0.' + strtrim(n, 2) + ')')
    if double(s) eq double(value) then return, s
  endfor
end

pro make_idl_references
  compile_opt idl2
  raster = 'iris_l2_20130902_182935_4000005156_raster_t000_r00000.fits'
  iris_burst_check, raster, threshold=40., output=output, /quiet
  ; An undefined threshold is replaced by the routine's default.
  iris_burst_check, raster, threshold=default_threshold, /quiet
  openw, lun, 'iris_burst_check_4000005156_r00000_thr40.ecsv', /get_lun
  printf, lun, format='(A)', [ $
    '# %ECSV 1.0', '# ---', '# datatype:', $
    '# - {name: step, datatype: int64}', '# - {name: y, datatype: int64}', '# - {name: group, datatype: int64}', $
    '# - {name: intensity, datatype: float64}', '# - {name: median, datatype: float64}', '# meta: !!omap', $
    "# - comment: ['iris_burst_check, file, threshold=40., output=output (IDL 9.2, SSW 2026-09-28) on', " + raster + ';', $
    '#       one row per burst pixel., step and y index the full file; the _si_iv_test.fits crop starts at y = y_offset.]', $
    '# - {threshold: 40.0}', '# - {default_threshold: ' + shortest(default_threshold) + '}', '# - {y_offset: 385}', $
    '# schema: astropy-2.0', 'step y group intensity median']
  foreach row, output do printf, lun, row.xpix, row.ypix, row.group, shortest(row.intensity), shortest(row.median), $
    format='(I0, " ", I0, " ", I0, " ", A, " ", A)'
  free_lun, lun

  iris_sji_burst_check, '2-sep-2013 17:00', output=output
  ; The frames of the _test.fits subset.
  frames = [0, 69, 227]
  openw, lun, 'iris_sji_burst_check_4000255147_pixels.ecsv', /get_lun
  printf, lun, format='(A)', [ $
    '# %ECSV 1.0', '# ---', '# datatype:', $
    '# - {name: frame, datatype: int64}', '# - {name: idl_frame, datatype: int64}', '# - {name: pixel, datatype: int64}', $
    '# - {name: group, datatype: int64}', '# - {name: intensity, datatype: float64}', '# meta: !!omap', $
    '# - comment: [Burst pixels of iris_sji_burst_check (IDL 9.2) for the frames in the _test.fits subset;, ' + $
    "'frame indexes the subset, idl_frame", $
    "#       the full file, pixel is the flat index of (y, x).']", $
    '# schema: astropy-2.0', 'frame idl_frame pixel group intensity']
  ; The file has 400 frames; frames without events are not in the output list.
  nevents = lonarr(400)
  npix = lonarr(400)
  foreach image, output do begin
    nevents[image.im_index] = image.nevents
    npix[image.im_index] = image.npix
    frame = where(frames eq image.im_index)
    if frame ge 0 then foreach pixel, image.index, i do printf, lun, frame, image.im_index, pixel, image.group[i], $
      shortest(image.int[i]), format='(I0, " ", I0, " ", I0, " ", I0, " ", A)'
  endforeach
  free_lun, lun

  openw, lun, 'iris_sji_burst_check_4000255147_summary.ecsv', /get_lun
  printf, lun, format='(A)', [ $
    '# %ECSV 1.0', '# ---', '# datatype:', $
    '# - {name: frame, datatype: int64}', '# - {name: nevents, datatype: int64}', '# - {name: npix, datatype: int64}', $
    '# meta: !!omap', $
    "# - comment: ['iris_sji_burst_check, ''2-sep-2013 17:00'', output=output (IDL 9.2, SSW 2026-09-28) on', " + $
    "'iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits:", $
    "#       events and burst pixels per frame.']", $
    '# schema: astropy-2.0', 'frame nevents npix']
  for frame = 0, 399 do printf, lun, frame, nevents[frame], npix[frame], format='(I0, " ", I0, " ", I0)'
  free_lun, lun
end
