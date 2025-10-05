/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.text;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import static java.text.DateFormatSymbols.*;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.Date;
    @Positive
import java.util.GregorianCalendar;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.SimpleTimeZone;
    @Positive
import java.util.SortedMap;
    @Positive
import java.util.TimeZone;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import sun.util.calendar.CalendarUtils;
    @Positive
import sun.util.calendar.ZoneInfoFile;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.TimeZoneNameUtility;

    @Positive
public class SimpleDateFormat extends DateFormat {

    @Positive
    public SimpleDateFormat() {
    @Positive
    }

    @Positive
    public SimpleDateFormat(String pattern) {
    @Positive
    }

    @Positive
    public SimpleDateFormat(String pattern, Locale locale) {
    @Positive
    }

    @Positive
    public SimpleDateFormat(String pattern, DateFormatSymbols formatSymbols) {
    @Positive
    }

    @Positive
    public void set2DigitYearStart(Date startDate);

    @Positive
    public Date get2DigitYearStart();

    @Positive
    @Override
    @Positive
    public StringBuffer format(Date date, StringBuffer toAppendTo, FieldPosition pos);

    @Positive
    @Override
    @Positive
    public AttributedCharacterIterator formatToCharacterIterator(Object obj);

    @Positive
    @Override
    @Positive
    public Date parse(String text, ParsePosition pos);

    @Positive
    public String toPattern();

    @Positive
    public String toLocalizedPattern();

    @Positive
    public void applyPattern(String pattern);

    @Positive
    public void applyLocalizedPattern(String pattern);

    @Positive
    public DateFormatSymbols getDateFormatSymbols();

    @Positive
    public void setDateFormatSymbols(DateFormatSymbols newFormatSymbols);

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}
