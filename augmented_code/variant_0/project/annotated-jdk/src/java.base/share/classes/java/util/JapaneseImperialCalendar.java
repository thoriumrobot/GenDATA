/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class JapaneseImperialCalendar {
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

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
import java.io.ObjectInputStream;
    @Positive
import sun.util.locale.provider.CalendarDataUtility;
    @Positive
import sun.util.calendar.BaseCalendar;
    @Positive
import sun.util.calendar.CalendarDate;
    @Positive
import sun.util.calendar.CalendarSystem;
    @Positive
import sun.util.calendar.CalendarUtils;
    @Positive
import sun.util.calendar.Era;
    @Positive
import sun.util.calendar.Gregorian;
    @Positive
import sun.util.calendar.LocalGregorianCalendar;
    @Positive
import sun.util.calendar.ZoneInfo;

    @Positive
class JapaneseImperialCalendar extends Calendar {

    @Positive
    public static final int BEFORE_MEIJI;

    @Positive
    public static final int MEIJI;

    @Positive
    public static final int TAISHO;

    @Positive
    public static final int SHOWA;

    @Positive
    public static final int HEISEI;

    @Positive
    @Override
    @Positive
    public String getCalendarType();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public void add(int field, int amount);

    @Positive
    @Override
    @Positive
    public void roll(int field, boolean up);

    @Positive
    @Override
    @Positive
    public void roll(int field, int amount);

    @Positive
    @Override
    @Positive
    public String getDisplayName(int field, int style, Locale locale);

    @Positive
    @Override
    @Positive
    public Map<String, Integer> getDisplayNames(int field, int style, Locale locale);

    @Positive
    public int getMinimum(int field);

    @Positive
    public int getMaximum(int field);

    @Positive
    public int getGreatestMinimum(int field);

    @Positive
    public int getLeastMaximum(int field);

    @Positive
    public int getActualMinimum(int field);

    @Positive
    public int getActualMaximum(int field);

    @Positive
    public Object clone();

    @Positive
    public TimeZone getTimeZone();

    @Positive
    public void setTimeZone(TimeZone zone);

    @Positive
    protected void computeFields();

    @Positive
    protected void computeTime();
    @Positive
}

}