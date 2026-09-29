; Creates the IDL references for irispy/utils/tests/test_mg_features.py (see overview.txt).
; Needs SolarSoft with this path:
;   !path = expand_path('+~/ssw/iris/idl') + ':' + expand_path('+~/ssw/gen/idl') + ':' + !path
; Run it in the directory holding iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits.

pro make_idl_references
  compile_opt idl2
  file = 'iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits'
  ; iris_get_mg_features_lev2 cannot open this file, as iris_obj's version check reads a C II window it lacks.
  ; This loop does what the wrapper does: descaled data, iris_data::getlam's wavelengths in nm, one step at a time.
  ; Window 0 covers only k and window 1 only h; /onlyk is not used, as it moves k to 279.644 nm.
  lc = replicate(!values.f_nan, 2, 2, 771, 3)
  bp = lc
  rp = lc
  for line = 0, 1 do begin
    spec = readfits(file, hdr, exten_no=line + 1)
    wave = (fxpar(hdr, 'CRVAL1') + (findgen(fxpar(hdr, 'NAXIS1')) + 1.0 - fxpar(hdr, 'CRPIX1')) $
      * fxpar(hdr, 'CDELT1')) / 10.
    for step = 0, 2 do begin
      iris_get_mg_features, spec[*, *, step], wave, [-40, 40], ltmp, rtmp, btmp
      lc[line, *, *, step] = ltmp[line, *, *]
      bp[line, *, *, step] = btmp[line, *, *]
      rp[line, *, *, step] = rtmp[line, *, *]
    endfor
  endfor
  save, lc, bp, rp, filename='iris_get_mg_features_lev2_4000005156_r00000.sav'

  ; Evenly spaced knots, sampled every 0.05 (t is built from 2.5 downwards and reversed).
  x = dindgen(21) / 4 - 2.5
  y = exp(-x^2) + 0.1d * x
  t = reverse(2.5 - dindgen(101) / 100 * 5)
  tension_0 = spline(x, y, t, 0.)
  tension_1 = spline(x, y, t, 1.)
  ; Unevenly spaced knots: lib/spline.pro solves its last row with the first interval's diagonal.
  uneven_x = [0d, 1, 2, 3, 4, 10]
  uneven_y = [0d, 1, 0, 1, 0, 1]
  uneven_t = [3.5d, 5, 7, 9]
  uneven_tension_0 = spline(uneven_x, uneven_y, uneven_t, 0.)
  uneven_tension_1 = spline(uneven_x, uneven_y, uneven_t, 1.)
  save, x, y, t, tension_0, tension_1, uneven_x, uneven_y, uneven_t, uneven_tension_0, uneven_tension_1, $
    filename='idl_spline.sav'
end
