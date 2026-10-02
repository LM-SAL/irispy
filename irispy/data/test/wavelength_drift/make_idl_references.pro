; Creates the IDL reference tables for irispy/utils/tests/test_wavelength_drift.py (see overview.txt).
; Needs SolarSoft with $SSW_IRIS set and this path (read_iris_l2 calls create_struct_temp from vobs/ontology):
;   !path = expand_path('+~/ssw/iris/idl') + ':' + expand_path('+~/ssw/gen/idl') + ':' $
;     + expand_path('+~/ssw/vobs/ontology/idl') + ':' + !path
; Run it in a directory holding iris_l2_20130902_182935_4000005156_raster_t000_r00000.fits and
; iris_l2_20140708_114109_3824262996_raster_t000_r00000.fits.

; The shortest string that reads back as the same double, written as Python's repr (and so astropy) writes it.
function shortest, value
  compile_opt idl2
  if ~finite(value) then return, 'nan'
  for n = 1, 16 do begin
    s = string(value, format='(E0.' + strtrim(n, 2) + ')')
    if double(s) eq value then break
  endfor
  parts = strsplit(s, 'E', /extract)
  exponent = long(parts[1])
  digits = strjoin(strsplit(parts[0], '-.', /extract))
  while strlen(digits) gt 1 && strmid(digits, strlen(digits) - 1) eq '0' do digits = strmid(digits, 0, strlen(digits) - 1)
  sign = value lt 0 ? '-' : ''
  if exponent lt -4 || exponent ge 16 then begin
    if strlen(digits) gt 1 then digits = strmid(digits, 0, 1) + '.' + strmid(digits, 1)
    return, sign + digits + 'e' + (exponent lt 0 ? '-' : '+') + string(abs(exponent), format='(I02)')
  endif
  if exponent lt 0 then return, sign + '0.' + strmid('000', 0, -exponent - 1) + digits
  fraction = strmid(digits, exponent + 1)
  return, sign + strmid(digits + '000000000000000', 0, exponent + 1) + '.' + (fraction eq '' ? '0' : fraction)
end

; Writes the per-step shifts and the drifts of iris_prep_wavecorr_l2 on one raster file, in time order.
pro write_reference, raster, output
  compile_opt idl2
  r = iris_prep_wavecorr_l2(raster)
  columns = ['Ni I', 'Mn I', 'Fe I', 'O I', 'Fe II', 'nuv', 'fuv']
  header = ['# %ECSV 1.0', '# ---', '# datatype:', '# - {name: time, datatype: string}', $
    '# - {name: ' + columns + ', unit: Angstrom, datatype: float64}', '# meta: !!omap', $
    "# - comment: ['r = iris_prep_wavecorr_l2(file) (IDL 9.2, SSW 2026-09-28) on', '" + raster + ":',", $
    "#     'per-step shifts r.corrs of the 5 reference lines and the drifts r.corr_nuv, r.corr_fuv.']", $
    '# - __serialized_columns__:']
  foreach column, columns[sort(columns)], i do header = [header, '#     ' + column + ':', $
    '#       __class__: astropy.units.quantity.Quantity', $
    '#       unit: ' + (i eq 0 ? '&id001 !astropy.units.Unit {unit: Angstrom}' : '*id001'), $
    '#       value: !astropy.table.SerializedColumn {name: ' + column + '}']
  header = [header, '#     time:', '#       __class__: astropy.time.core.Time', '#       format: isot', $
    "#       in_subfmt: '*'", "#       out_subfmt: '*'", '#       precision: 3', '#       scale: utc', $
    '#       value: !astropy.table.SerializedColumn {name: time}', '# schema: astropy-2.0', $
    'time "Ni I" "Mn I" "Fe I" "O I" "Fe II" nuv fuv']
  openw, lun, output, /get_lun
  printf, lun, header, format='(A)'
  ; corr_tai, corr_nuv and corr_fuv are sorted by time; corrs and times are not.
  order = sort(r.tais[0, *])
  for i = 0, n_elements(order) - 1 do begin
    values = [reform(r.corrs[0, order[i], *]), r.corr_nuv[i], r.corr_fuv[i]]
    row = r.times[0, order[i]]
    foreach value, values do row += ' ' + shortest(value)
    printf, lun, row, format='(A)'
  endfor
  free_lun, lun
end

pro make_idl_references
  compile_opt idl2
  foreach raster, ['iris_l2_20130902_182935_4000005156', 'iris_l2_20140708_114109_3824262996'] do $
    write_reference, raster + '_raster_t000_r00000.fits', $
      'iris_prep_wavecorr_l2_' + (strsplit(raster, '_', /extract))[-1] + '_r00000.ecsv'
end
