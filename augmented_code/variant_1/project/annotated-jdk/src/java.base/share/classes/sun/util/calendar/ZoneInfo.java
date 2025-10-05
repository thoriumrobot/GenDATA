/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.calendar;

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
import java.util.Date;
    @Positive
import java.util.Map;
    @Positive
import java.util.SimpleTimeZone;
    @Positive
import java.util.TimeZone;

    @Positive
public class ZoneInfo extends TimeZone {

    @Positive
    public ZoneInfo() {
    @Positive
    }

    @Positive
    public ZoneInfo(String ID, int rawOffset) {
    @Positive
    }

    @Positive
    public int getOffset(long date);

    @Positive
    public int getOffsets(long utc, int[] offsets);

    @Positive
    public int getOffsetsByStandard(long standard, int[] offsets);

    @Positive
    public int getOffsetsByWall(long wall, int[] offsets);

    @Positive
    public int getOffset(int era, int year, int month, int day, int dayOfWeek, int milliseconds);

    @Positive
    public synchronized void setRawOffset(int offsetMillis);

    @Positive
    public int getRawOffset();

    @Positive
    public boolean isDirty();

    @Positive
    public boolean useDaylightTime();

    @Positive
    @Override
    @Positive
    public boolean observesDaylightTime();

    @Positive
    public boolean inDaylightTime(Date date);

    @Positive
    public int getDSTSavings();

    @Positive
    public String toString();

    @Positive
    public static String[] getAvailableIDs();

    @Positive
    public static String[] getAvailableIDs(int rawOffset);

    @Positive
    public static TimeZone getTimeZone(String ID);

    @Positive
    public SimpleTimeZone getLastRuleInstance();

    @Positive
    public Object clone();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public boolean hasSameRules(TimeZone other);

    @Positive
    public static Map<String, String> getAliasTable();
    @Positive
}
