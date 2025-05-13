;VERSION 0.1

;Author : Vaishali S.,CLASS team

;MODIFICATION HISTORY
;24th Dec 2020 : Created.

;PURPOSE:
;Program to add specified no. of L1 files based on user provided start and end times
;Assumed that the time interval provided by the user does not have any time gaps
;Output added files are stored in a sub directory named 'L1_ADDED_FILES_TIME' 
;within the same directory where the input files are found

;Functions / Procedures in the code:
;convert_time_str_format
;utc_to_seconds
;write_summed_fits
;CLASS_add_L1_files_time.pro - main program

;USAGE :
;CLASS_ADD_L1_FILES_TIME, '/path/to/directory/having/input/L1_FITS_files/', 'yyyy-mm-ddTHH:MM:SS.MSE', 'yyyy-mm-ddTHH:MM:SS.MSE', '/path/to/directory/to/store/output/added_L1_FITS_files/'
;
;in the above, the time arguments are to be entered as in the following example
;CLASS_ADD_L1_FILES_TIME, '/path/to/directory/having/input/L1_FITS_files/' , '2019-11-18T23:59:41.268', '2019-11-19T00:00:29.267', '/path/to/directory/to/store/output/added_L1_FITS_files/'

;PURPOSE:
;to convert the file times string to a format expected by another function

FUNCTION CONVERT_TIME_STR_FORMAT, time_arr
yr1 = strmid(time_arr, 0, 4)
mon1 = strmid(time_arr, 4, 2)
date1 = strmid(time_arr, 6, 2)
hr1 = strmid(time_arr, 9, 2)
min1 = strmid(time_arr, 11, 2)
sec1 = strmid(time_arr, 13, 2)
mse1 = strmid(time_arr, 15, 3)

UTC_all = yr1 + "-" + mon1 + "-" + date1 + "T" + hr1 + ":" + min1 + ":" + sec1 + "." + mse1

return, UTC_all

END

;PURPOSE:
;to convert time in UTC to seconds

FUNCTION UTC_TO_SECONDS, utc

;obtaining the seconds elapsed since Jan 01, 1970
secperday = long(24L * 60L * 60L)
jul_day_first = julday(1, 1, 1970, 00, 00, 00)
;jul_day_first_sec =  jul_day_first * secperday
;print, 'Space craft zero time in seconds since Jan 01, 1970 : ', jul_day_sp_zero_sec

;for input UTC, get the seconds since 01-01-1970
input_UTC = strsplit(UTC, '-T:', /extract)
;print, input_UTC
year = input_UTC[0]
month = input_UTC[1]
date = input_UTC[2]
hour = input_UTC[3]
min = input_UTC[4]
sec = input_UTC[5]
jul_day_for_input_UTC = julday(month, date, year, hour, min, sec)
input_UTC_in_seconds = (jul_day_for_input_UTC - jul_day_first) * secperday
;print, 'Input UTC time in seconds since Jan 01, 1970 : ', input_UTC_in_seconds

return, input_UTC_in_seconds

END

;PURPOSE:
;writes FITS files for the specified SCD's summed data
;INPUTS:
;summed_SCD_opfname : Output FITS file name along with full path
;data_SCD_summed : data to be written to FITS file
;ip_file : Input data filename
;dataset_start : Dataset no. from which adding started
;dataset_end : Dataset no. at which adding ended
;OUTPUTS:
;FITS file written to disk.

PRO WRITE_SUMMED_FITS, summed_SCD_opfname, data_SCD_summed, ip_file, expotime, starttime, endtime, SPICE_val

;print, 'writing PHDU'
fxhmake, pheader, /extend
fxaddpar, pheader, 'DATE', systime(0), 'file creation date'
fxwrite, summed_SCD_opfname, pheader

;now write BT
;BT header
noofrows1 = n_elements(data_SCD_summed)
fxbhmake,btheader1, noofrows1

;BT data - channels, hist_8s_shifted[channels]
channels = indgen(n_elements(data_SCD_summed))   ;;;should be 2048
;create columns - using one data from each column as sample
fxbaddcol, col1, btheader1, channels(0)
fxbaddcol, col2, btheader1, data_SCD_summed(0)

fxaddpar, btheader1, 'TTYPE1', 'CHANNEL', 'PHA channel'
fxaddpar, btheader1, 'TTYPE2', 'COUNTS', 'Counts per channel'
fxaddpar, btheader1, 'TUNIT2', 'count', 'Physical unit of field'
fxaddpar, btheader1, 'EXTNAME', 'SPECTRUM', 'Name of binary table extension'
fxaddpar, btheader1, 'HDUCLASS', 'OGIP', 'format conforms to OGIP standard'
fxaddpar, btheader1, 'HDUCLAS1', 'SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)'
fxaddpar, btheader1, 'HDUVERS1', '1.1.0', 'Obsolete - included for backwards compatibility'
fxaddpar, btheader1, 'HDUVERS', '1.1.0', 'Version of format (OGIP memo OGIP-92-007a)'
fxaddpar, btheader1, 'HDUCLAS2', 'UNKNOWN', 'Maybe TOTAL, NET or BKG Spectrum'
fxaddpar, btheader1, 'HDUCLAS3', 'COUNT', 'PHA data stored as Counts (not count/s)'
fxaddpar, btheader1, 'TLMIN1', 0, 'Lowest legal channel number'
fxaddpar, btheader1, 'TLMAX1', 2047, 'Highest legal channel number'

fxaddpar, btheader1, 'TELESCOP', 'CHANDRAYAAN-2', 'mission/satellite name'
fxaddpar, btheader1, 'INSTRUME', 'CLASS', 'instrument/detector name'
fxaddpar, btheader1, 'FILTER', 'none', 'filter in use'
;fxaddpar, btheader1, 'EXPOSURE', 3.200000E+01, 'exposure (in seconds)'
fxaddpar, btheader1, 'EXPOSURE', expotime, 'exposure (in seconds)'
fxaddpar, btheader1, 'AREASCAL', 1.000000E+00, 'area scaling factor'
fxaddpar, btheader1, 'BACKFILE', 'NONE', 'associated background filename'
fxaddpar, btheader1, 'BACKSCAL', 1.000000E+00, 'background file scaling factor'
fxaddpar, btheader1, 'CORRFILE', 'NONE', 'associated correction filename'
fxaddpar, btheader1, 'CORRSCAL', 1.000000E+00, 'correction file scaling factor'
fxaddpar, btheader1, 'RESPFILE', 'NONE', 'associated redistrib matrix filename'
fxaddpar, btheader1, 'ANCRFILE', 'NONE', 'associated ancillary response filename'
fxaddpar, btheader1, 'PHAVERSN', '1992a', 'obsolete'
fxaddpar, btheader1, 'DETCHANS', 2048, 'total number possible channels'
fxaddpar, btheader1, 'CHANTYPE', 'PHA', 'channel type (PHA, PI etc)'
fxaddpar, btheader1, 'POISSERR', 'T', 'Poissonian errors to be assumed'
fxaddpar, btheader1, 'STAT_ERR', 0, 'no statistical error specified'
fxaddpar, btheader1, 'SYS_ERR', 0, 'no systematic error specified'
fxaddpar, btheader1, 'GROUPING', 0, 'no grouping of the data has been defined'
fxaddpar, btheader1, 'QUALITY', 0, 'no data quality information specified'
;fxaddpar, btheader1, 'HISTORY infile :SCD03_72_73_shifted.txt
fxaddpar, btheader1, 'EQUINOX', 2.0000E+03, 'Equinox of Celestial coord system'
fxaddpar, btheader1, 'DATE', systime(0), 'file creation date'
;fxaddpar, btheader1, 'EXPTIME', '8 seconds', 'Exposure Time'
fxaddpar, btheader1, 'PROGRAM', 'CLASS_add_scds_time.pro', 'Program that created the file'
fxaddpar, btheader1, 'IPFILE', ip_file, 'Input file name'
;fxaddpar, btheader1, 'STARTSET', dataset_start, 'Set no. from which adding started'
;fxaddpar, btheader1, 'ENDSET', dataset_end, 'Set no. at which adding ended'
fxaddpar, btheader1, 'STARTIME', starttime[0], 'Start time in UTC'
fxaddpar, btheader1, 'ENDTIME', endtime[0], 'End time in UTC'
;SPICE parameters
fxaddpar, btheader1, 'SAT_ALT', round(SPICE_val.sat_alt_mean * 10000.0) / 10000.0, 'Sub-satellite point altitude - Mean value for the added files'
fxaddpar, btheader1, 'V0_LAT', round(SPICE_val.v0_lat * 10000.0) / 10000.0, 'Pixel corner 0 latitude (deg)'
fxaddpar, btheader1, 'V1_LAT', round(SPICE_val.v1_lat * 10000.0) / 10000.0, 'Pixel corner 1 latitude (deg)'
fxaddpar, btheader1, 'V2_LAT', round(SPICE_val.v2_lat * 10000.0) / 10000.0, 'Pixel corner 2 latitude (deg)'
fxaddpar, btheader1, 'V3_LAT', round(SPICE_val.v3_lat * 10000.0) / 10000.0, 'Pixel corner 3 latitude (deg)'
fxaddpar, btheader1, 'V0_LON', round(SPICE_val.v0_lon * 10000.0) / 10000.0, 'Pixel corner 0 longitude (deg)'
fxaddpar, btheader1, 'V1_LON', round(SPICE_val.v1_lon * 10000.0) / 10000.0, 'Pixel corner 1 longitude (deg)'
fxaddpar, btheader1, 'V2_LON', round(SPICE_val.v2_lon * 10000.0) / 10000.0, 'Pixel corner 2 longitude (deg)'
fxaddpar, btheader1, 'V3_LON', round(SPICE_val.v3_lon * 10000.0) / 10000.0, 'Pixel corner 3 longitude (deg)'
fxaddpar, btheader1, 'SOLARANG', round(SPICE_val.sol_ang_mean * 10000.0) / 10000.0, 'Angle between surface normal,sun vector (deg)';Mean value for the added files
fxaddpar, btheader1, 'PHASEANG', round(SPICE_val.phs_ang_mean * 10000.0) / 10000.0, 'Angle between boresight vector,sun vector (deg)';Mean value for the added files
fxaddpar, btheader1, 'EMISNANG', SPICE_val.emi_ang_mean, 'Angle btwn boresight vector,emitted X-rays(deg)';Mean value for the added files

;fxaddpar, btheader1,
;fxaddpar, btheader1,

;rewrite btheader to accommodate the new columns
;print, 'fxbcreate'
fxbcreate, funit1, summed_SCD_opfname, btheader1

;write the columns to the file
;print, 'fxbwrite1'
for k = 1, noofrows1 do begin
   fxbwrite, funit1, channels(k - 1), col1, k   ;channels
   fxbwrite, funit1, data_SCD_summed(k - 1), col2, k   ;counts
endfor

;print, 'writing first extension over'
;close
;print, 'fxbfinish'
fxbfinish, funit1

END


;PURPOSE:
;mentioned at the start of the file

;INPUTS:
;ipfile_dir         : directory containing added L1 FITS files
;UTC_start_user     : start time in UTC, provided by user, from which files 
;                     should be added - format : YYYY-MM-DDThh:mm:ss.mse
;UTC_end_user       : end time in UTC, provided by user, from which files should 
;                     be added, format same as above
;op_dir_location    : directory location to store the output files directory


;OUTPUTS:
;FITS file written to disk.

PRO CLASS_ADD_L1_FILES_TIME, ipfile_dir, UTC_start_user, UTC_end_user, op_dir_location

version = '0.1'

;LOG FILE NEEDED??

;carry out checks on the command line arguments

if (ipfile_dir eq '') OR (UTC_start_user eq '') OR (UTC_end_user eq '') OR (op_dir_location eq '') then begin

   print, "Please enter all the input values."
   print, "Usage : "
   print, "CLASS_ADD_L1_FILES_TIME, '/path/to/directory/having/input/L1_FITS_files/', 'yyyy-mm-ddTHH:MM:SS.MSE', 'yyyy-mm-ddTHH:MM:SS.MSE', '/home/vaishali/WORK/CLASS_DATA/CLASS_L0_DATA/CLAXXD18CHO0003803NNNN19237070348454_V1_0/Outputs_CLA01D18CHO0003803016019237070348454_00.pld/ADD_SCD/'"
endif else begin
   print, ''
   print, 'USER INPUTS'
   print, '-----------'
   print, 'Input files directory : ', ipfile_dir
   print, 'Start UTC : ', UTC_start_user
   print, 'End UTC : ', UTC_end_user
   print, 'Location to store output files : ', op_dir_location
   print, ''

   path_separator = path_sep() ;for appending to op_subdir_name

   ;check if the input and output directories ends with a /
   ;add a '/' at the end even if it is there or not - does not cause any issue
   ipfile_dir = ipfile_dir + path_separator
   op_dir_location = op_dir_location + path_separator

   fname_ext = '.fits'

   ;create array to hold the complete data of all files to be added
   data_SCD_summed = fltarr(2048)
   ip_file = '';got later

   ;check if a subdirectory to store output files exists in ipfile_dir
   op_subdir_name = 'L1_ADDED_FILES_TIME'

   ;if the above subdir exists then use the same else create it newly
   subdir_exists = file_test(op_dir_location + op_subdir_name)
   if (subdir_exists eq 1) then begin
      print, 'Subdirectory (L1_ADDED_FILES_TIME) to store output files exists.'
   endif else begin
      print, 'Subdirectory (L1_ADDED_FILES_TIME) to store output files does not exist. Will create now.'
      ;make a directory to store the output files - in a sub directory where the
      ;input data files are found
      cd, op_dir_location, current = program_dir
      file_mkdir, op_subdir_name
      cd, program_dir
   endelse

   op_subdir_name_full = op_dir_location + op_subdir_name + path_separator
   print, ''

   ;convert the input start and end UTC into seconds
   UTC_start_user_secs = utc_to_seconds(UTC_start_user)
   UTC_end_user_secs = utc_to_seconds(UTC_end_user)
   print, ''
   ;print, 'Start UTC in seconds : ', UTC_start_user_secs
   ;print, 'End UTC in seconds : ', UTC_end_user_secs
   print, ''

   ;read all the L1 files in the specified input directory and get the file times
   ; and write out as a text file ??
   file_list = file_search(ipfile_dir + "*.fits", count = n_files_in_ipdir)   ;o/p is an array;STOP
   ;above array is sorted
   ;get only the filenames (remove directory details)
   ;ch2_cla_l1_20191118T235925268_20191118T235933268.fits
   file_list_basename = file_basename(file_list) ;;;this too is sorted
   ;get only the time string
   ;20191118T235925268_20191118T235933268
   file_time_only = strmid(file_list_basename, 11, 37)
   ;get the start and end times separately
   file_start_times = strmid(file_time_only, 0, 18)
   file_end_times = strmid(file_time_only, 19, 18)

   ;convert into format needed by utc_to_seconds
   ;2019-11-18T23:59:25.268
   startUTC_arr = convert_time_str_format(file_start_times)
   endUTC_arr = convert_time_str_format(file_end_times)

   ;now get the start and end UTC from file in seconds
   start_UTC_arr_secs = dblarr(n_elements(startUTC_arr))
   end_UTC_arr_secs = dblarr(n_elements(endUTC_arr))
   for i = 0, n_elements(startUTC_arr) -1 do begin
      start_UTC_arr_secs[i] = utc_to_seconds(startUTC_arr[i])
      end_UTC_arr_secs[i] = utc_to_seconds(endUTC_arr[i])
   endfor

   ;now for the input start and end UTC find the nearest values
   ind_UTC_match = where(start_UTC_arr_secs ge UTC_start_user_secs AND end_UTC_arr_secs le UTC_end_user_secs, count_ind_UTC_match)

   if (count_ind_UTC_match gt 0) then begin

      ;get the first index from start and last index from end
      start_index = ind_UTC_match[0]
      end_index = ind_UTC_match[count_ind_UTC_match - 1]

      ;get the corresponding matching UTC values
      startUTC_match_arr = startUTC_arr[ind_UTC_match]
      endUTC_match_arr = endUTC_arr[ind_UTC_match]

      UTC_start_actual = startUTC_match_arr[0]
      UTC_end_actual = endUTC_match_arr[count_ind_UTC_match - 1]

      ;make the UTC values in the format of add SCD o/p FITS files 
      ;eg. ch2_cla_l1_20191129T003346791_20191129T003354791.fits
      ;now for the above window numbers, get those files only to add further
      ;since file naming convention is known, can easily frame the file names
      files_to_add_arr = strarr(count_ind_UTC_match)
      ;arrays to hold SPICE parameters retrieved from the headers of files 
      ;to be added
      sat_alt_all = dblarr(count_ind_UTC_match)
      sol_ang_all = dblarr(count_ind_UTC_match)
      phs_ang_all = dblarr(count_ind_UTC_match)
      emi_ang_all = dblarr(count_ind_UTC_match)

      ;for the corner coordinates obtained from first file
      v0_lat1 = 0.0D
      v1_lat1 = 0.0D
      v2_lat1 = 0.0D
      v3_lat1 = 0.0D
      v0_lon1 = 0.0D
      v1_lon1 = 0.0D
      v2_lon1 = 0.0D
      v3_lon1 = 0.0D

      ;for the corner coordinates obtained from last file
      v0_lat2 = 0.0D
      v1_lat2 = 0.0D
      v2_lat2 = 0.0D
      v3_lat2 = 0.0D
      v0_lon2 = 0.0D
      v1_lon2 = 0.0D
      v2_lon2 = 0.0D
      v3_lon2 = 0.0D

      for j = 0, count_ind_UTC_match - 1 do begin
         start_UTC_split = strsplit(startUTC_match_arr[j], '-:.', /extract)
         start_UTC_str = start_UTC_split[0] + start_UTC_split[1] + start_UTC_split[2] + start_UTC_split[3] + start_UTC_split[4] + start_UTC_split[5]
         end_UTC_split = strsplit(endUTC_match_arr[j], '-:.', /extract)
         end_UTC_str = end_UTC_split[0] + end_UTC_split[1] + end_UTC_split[2] + end_UTC_split[3] + end_UTC_split[4] + end_UTC_split[5]
         files_to_add_arr[j] = 'ch2_cla_l1_' +  start_UTC_str + '_' + end_UTC_str + fname_ext

         file = file_search(ipfile_dir + files_to_add_arr[j])   ;o/p is an array;STOP
         n_files_found = n_elements(file)
;         print, 'No. of files found : ', n_files_found;should be 1 always
;         print, file

         ;read the fits file here - 2 cols - first channel no second counts
;         print, 'First extension'
         fxbopen, unit1, file, 1
         hdr = fxbheader(unit1)
         ;get the required parameters from the header
         ip_file = fxpar(hdr, 'IPFILE')
         sat_alt_all[j] = fxpar(hdr, 'SAT_ALT')
         sol_ang_all[j] = fxpar(hdr, 'SOLARANG')
         phs_ang_all[j] = fxpar(hdr, 'PHASEANG')
         emi_ang_all[j] = fxpar(hdr, 'EMISNANG')
         ;from the first and last files, get the v0, v1,v2, v3 values
         ;so totally 16 values
         if (j eq 0) then begin
;            print, 'first file ', j
            v0_lat1 = fxpar(hdr, 'V0_LAT')
            v1_lat1 = fxpar(hdr, 'V1_LAT')
            v2_lat1 = fxpar(hdr, 'V2_LAT')
            v3_lat1 = fxpar(hdr, 'V3_LAT')
            v0_lon1 = fxpar(hdr, 'V0_LON')
            v1_lon1 = fxpar(hdr, 'V1_LON')
            v2_lon1 = fxpar(hdr, 'V2_LON')
            v3_lon1 = fxpar(hdr, 'V3_LON')
         endif else if (j eq count_ind_UTC_match - 1) then begin
;            print, 'last file', j
            v0_lat2 = fxpar(hdr, 'V0_LAT')
            v1_lat2 = fxpar(hdr, 'V1_LAT')
            v2_lat2 = fxpar(hdr, 'V2_LAT')
            v3_lat2 = fxpar(hdr, 'V3_LAT')
            v0_lon2 = fxpar(hdr, 'V0_LON')
            v1_lon2 = fxpar(hdr, 'V1_LON')
            v2_lon2 = fxpar(hdr, 'V2_LON')
            v3_lon2 = fxpar(hdr, 'V3_LON')
         endif
   
         ;fxbhelp, unit1
         fxbread, unit1, channel_no, 1
         fxbread, unit1, counts, 2
         fxbclose, unit1
;         help, channel_no, counts
;         print, counts[440]
         data_SCD_summed[*] += counts

      endfor

      ;calculating the value of SPICE parameters for the added files
      lon1 = [v0_lon1, v1_lon1, v2_lon1, v3_lon1]
      lon2 = [v0_lon2, v1_lon2, v2_lon2, v3_lon2]
      lat1 = [v0_lat1, v1_lat1, v2_lat1, v3_lat1]
      lat2 = [v0_lat2, v1_lat2, v2_lat2, v3_lat2]
      ;print, 'lon1, lon2, lat1, lat2'
      ;print, lon1, lon2, lat1, lat2
      dy1 = lat1
      dx1 = lon1
      dy2 = lat2
      dx2 = lon2

      ;defining cx and cy - else error shown when above values are 0 (CHECK)
      cx = dblarr(4)
      cy = dblarr(4)

      ;ADDING the following from Netra's code
      if (max(dy1) GT max(dy2)) then begin
         ;first footprint is at top
         cx = [dx1(0), dx2(1), dx2(2), dx1(3)]   ;all final vertices' longitude
         cy = [dy1(0), dy2(1), dy2(2), dy1(3)]   ;all final vertices' latitude
      endif else if (max(dy1) LT max(dy2)) then begin   ; second footprint is at top
         cx = [ dx2(0), dx1(1), dx1(2), dx2(3)]   ;lon
         cy = [ dy2(0), dy1(1), dy1(2), dy2(3)]   ;lat
      endif

      ;print, 'Longitude values for the final coordinates : ', cx
      ;print, 'Latitude values for the final coordinates : ', cy

      ;print, sat_alt_all
      sat_alt_mean = mean(sat_alt_all)
      sol_ang_mean = mean(sol_ang_all)
      phs_ang_mean = mean(phs_ang_all)
      emi_ang_mean = mean(emi_ang_all)
      ;print, sat_alt_mean, sol_ang_mean, phs_ang_mean, emi_ang_mean

      ;put all the SPICE values in a structure
      SPICE_val = {                            $
                  sat_alt_mean : sat_alt_mean, $
                  sol_ang_mean : sol_ang_mean, $
                  phs_ang_mean : phs_ang_mean, $
                  emi_ang_mean : emi_ang_mean, $
                  v0_lat       : cy[0],        $
                  v1_lat       : cy[1],        $
                  v2_lat       : cy[2],        $
                  v3_lat       : cy[3],        $
                  v0_lon       : cx[0],        $
                  v1_lon       : cx[1],        $
                  v2_lon       : cx[2],        $
                  v3_lon       : cx[3]         $
                  }

      ;print, ''
      print, 'Total no. of files matching the start and end times provided by user : ', count_ind_UTC_match
      print, 'List of files added : '
      print, files_to_add_arr

      print, ''
      print, 'Start UTC provided by user : ', UTC_start_user
      print, 'End UTC provided by user : ', UTC_end_user


      ;now write as a FITS file
      ;making the input start and end UTC string format compatible for adding to 
      ;output filename - SHUOULD USE ACTUAL START AND END TIMES
      UTC_start_split = strsplit(UTC_start_actual, '-:.', /extract)
      UTC_start_str = UTC_start_split[0] + UTC_start_split[1] + UTC_start_split[2] + UTC_start_split[3] + UTC_start_split[4] + UTC_start_split[5]
      UTC_end_split = strsplit(UTC_end_actual, '-:.', /extract)
      UTC_end_str = UTC_end_split[0] + UTC_end_split[1] + UTC_end_split[2] + UTC_end_split[3] + UTC_end_split[4] + UTC_end_split[5]

      summed_SCD_opfname = op_subdir_name_full + 'ch2_cla_L1_time_added_' + UTC_start_str + '-' + UTC_end_str + fname_ext

      ;calculate the exposure time based on the no. of files added - each file is 8s
      expotime = count_ind_UTC_match * 8
      print, 'Exposure time (seconds): ', expotime

      ;write a fits file
      write_summed_fits, summed_SCD_opfname, data_SCD_summed, ip_file, expotime, UTC_start_actual, UTC_end_actual, SPICE_val
      print, ''
      print, '***************'

      print, 'Program CLASS_add_L1_files_time.pro completed.'
      print, 'Output file name : ', summed_SCD_opfname
      print, 'Output files of this program are stored in the following location : '
      print, op_subdir_name_full

   endif else begin
 
      print, "Could not find files to add for the user specified start and end times."
      print, "Please try again with proper values."

   endelse
endelse

END

