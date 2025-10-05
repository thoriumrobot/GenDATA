/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
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
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.text.spi.DateFormatSymbolsProvider;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import sun.util.locale.provider.CalendarDataUtility;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;
    @Positive
import sun.util.locale.provider.TimeZoneNameUtility;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public class DateFormatSymbols implements Serializable, Cloneable {

    @Positive
    public DateFormatSymbols() {
    @Positive
    }

    @Positive
    public DateFormatSymbols(Locale locale) {
    @Positive
    }

    @Positive
    public static Locale[] getAvailableLocales();

    @Positive
    public static final DateFormatSymbols getInstance();

    @Positive
    public static final DateFormatSymbols getInstance(Locale locale);

    @Positive
    static final DateFormatSymbols getInstanceRef(Locale locale);

    @Positive
    public String @ArrayLen(2) [] getEras();

    @Positive
    public void setEras(String @ArrayLen(2) [] newEras);

    @Positive
    public String @ArrayLen({ 12, 13 }) [] getMonths();

    @Positive
    public void setMonths(String @ArrayLen({ 12, 13 }) [] newMonths);

    @Positive
    public String @ArrayLen(13) [] getShortMonths();

    @Positive
    public void setShortMonths(String @ArrayLen(13) [] newShortMonths);

    @Positive
    public String @ArrayLen(8) [] getWeekdays();

    @Positive
    public void setWeekdays(String @ArrayLen(8) [] newWeekdays);

    @Positive
    public String @ArrayLen(8) [] getShortWeekdays();

    @Positive
    public void setShortWeekdays(String @ArrayLen(8) [] newShortWeekdays);

    @Positive
    public String @ArrayLen(2) [] getAmPmStrings();

    @Positive
    public void setAmPmStrings(String @ArrayLen(2) [] newAmpms);

    @Positive
    public String[] @MinLen(5) [] getZoneStrings();

    @Positive
    public void setZoneStrings(String[] @MinLen(5) [] newZoneStrings);

    @Positive
    public String getLocalPatternChars();

    @Positive
    public void setLocalPatternChars(String newLocalPatternChars);

    @Positive
    public Object clone();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    final int getZoneIndex(String ID);

    @Positive
    final String[][] getZoneStringsWrapper();
    @Positive
}
