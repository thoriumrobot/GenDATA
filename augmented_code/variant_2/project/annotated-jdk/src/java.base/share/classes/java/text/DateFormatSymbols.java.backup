/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.ArrayLen;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.ref.SoftReference;
import java.text.spi.DateFormatSymbolsProvider;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import sun.util.locale.provider.CalendarDataUtility;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;
import sun.util.locale.provider.ResourceBundleBasedAdapter;
import sun.util.locale.provider.TimeZoneNameUtility;

@AnnotatedFor({ "index" })
public class DateFormatSymbols implements Serializable, Cloneable {

    public DateFormatSymbols() {
    }

    public DateFormatSymbols(Locale locale) {
    }

    public static Locale[] getAvailableLocales();

    public static final DateFormatSymbols getInstance();

    public static final DateFormatSymbols getInstance(Locale locale);

    static final DateFormatSymbols getInstanceRef(Locale locale);

    public String @ArrayLen(2) [] getEras();

    public void setEras(String @ArrayLen(2) [] newEras);

    public String @ArrayLen({ 12, 13 }) [] getMonths();

    public void setMonths(String @ArrayLen({ 12, 13 }) [] newMonths);

    public String @ArrayLen(13) [] getShortMonths();

    public void setShortMonths(String @ArrayLen(13) [] newShortMonths);

    public String @ArrayLen(8) [] getWeekdays();

    public void setWeekdays(String @ArrayLen(8) [] newWeekdays);

    public String @ArrayLen(8) [] getShortWeekdays();

    public void setShortWeekdays(String @ArrayLen(8) [] newShortWeekdays);

    public String @ArrayLen(2) [] getAmPmStrings();

    public void setAmPmStrings(String @ArrayLen(2) [] newAmpms);

    public String[] @MinLen(5) [] getZoneStrings();

    public void setZoneStrings(String[] @MinLen(5) [] newZoneStrings);

    public String getLocalPatternChars();

    public void setLocalPatternChars(String newLocalPatternChars);

    public Object clone();

    @Override
    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    final int getZoneIndex(String ID);

    final String[][] getZoneStringsWrapper();
}
