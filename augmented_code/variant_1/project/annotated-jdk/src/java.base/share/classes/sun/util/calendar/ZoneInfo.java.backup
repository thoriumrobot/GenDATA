/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.calendar;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Date;
import java.util.Map;
import java.util.SimpleTimeZone;
import java.util.TimeZone;

public class ZoneInfo extends TimeZone {

    public ZoneInfo() {
    }

    public ZoneInfo(String ID, int rawOffset) {
    }

    public int getOffset(long date);

    public int getOffsets(long utc, int[] offsets);

    public int getOffsetsByStandard(long standard, int[] offsets);

    public int getOffsetsByWall(long wall, int[] offsets);

    public int getOffset(int era, int year, int month, int day, int dayOfWeek, int milliseconds);

    public synchronized void setRawOffset(int offsetMillis);

    public int getRawOffset();

    public boolean isDirty();

    public boolean useDaylightTime();

    @Override
    public boolean observesDaylightTime();

    public boolean inDaylightTime(Date date);

    public int getDSTSavings();

    public String toString();

    public static String[] getAvailableIDs();

    public static String[] getAvailableIDs(int rawOffset);

    public static TimeZone getTimeZone(String ID);

    public SimpleTimeZone getLastRuleInstance();

    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public boolean hasSameRules(TimeZone other);

    public static Map<String, String> getAliasTable();
}
