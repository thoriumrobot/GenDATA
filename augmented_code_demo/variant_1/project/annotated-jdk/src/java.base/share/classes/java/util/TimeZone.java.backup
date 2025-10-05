/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.Serializable;
import java.time.ZoneId;
import jdk.internal.util.StaticProperty;
import sun.security.action.GetPropertyAction;
import sun.util.calendar.ZoneInfo;
import sun.util.calendar.ZoneInfoFile;
import sun.util.locale.provider.TimeZoneNameUtility;

@AnnotatedFor({ "lock", "nullness" })
public abstract class TimeZone implements Serializable, Cloneable {

    public TimeZone() {
    }

    public static final int SHORT;

    public static final int LONG;

    public abstract int getOffset(int era, int year, int month, int day, int dayOfWeek, int milliseconds);

    public int getOffset(long date);

    int getOffsets(long date, int[] offsets);

    public abstract void setRawOffset(@GuardSatisfied TimeZone this, int offsetMillis);

    public abstract int getRawOffset();

    public String getID();

    public void setID(@GuardSatisfied TimeZone this, String ID);

    @Pure
    public final String getDisplayName();

    @Pure
    public final String getDisplayName(Locale locale);

    @Pure
    public final String getDisplayName(boolean daylight, int style);

    @Pure
    public String getDisplayName(boolean daylight, int style, Locale locale);

    @Pure
    public int getDSTSavings();

    @Pure
    public abstract boolean useDaylightTime();

    public boolean observesDaylightTime();

    @Pure
    public abstract boolean inDaylightTime(Date date);

    @Pure
    public static synchronized TimeZone getTimeZone(String ID);

    public static TimeZone getTimeZone(ZoneId zoneId);

    public ZoneId toZoneId();

    public static synchronized String[] getAvailableIDs(int rawOffset);

    public static synchronized String[] getAvailableIDs();

    public static TimeZone getDefault();

    static TimeZone getDefaultRef();

    public static void setDefault(@Nullable TimeZone zone);

    public boolean hasSameRules(@Nullable TimeZone other);

    @SideEffectFree
    public Object clone(@GuardSatisfied TimeZone this);
}
