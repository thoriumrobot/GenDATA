/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.ZoneId;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.util.calendar.ZoneInfo;
    @Positive
import sun.util.calendar.ZoneInfoFile;
    @Positive
import sun.util.locale.provider.TimeZoneNameUtility;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public abstract class TimeZone implements Serializable, Cloneable {

    @Positive
    public TimeZone() {
    @Positive
    }

    @Positive
    public static final int SHORT;

    @Positive
    public static final int LONG;

    @Positive
    public abstract int getOffset(int era, int year, int month, int day, int dayOfWeek, int milliseconds);

    @Positive
    public int getOffset(long date);

    @Positive
    int getOffsets(long date, int[] offsets);

    @Positive
    public abstract void setRawOffset(@GuardSatisfied TimeZone this, int offsetMillis);

    @Positive
    public abstract int getRawOffset();

    @Positive
    public String getID();

    @Positive
    public void setID(@GuardSatisfied TimeZone this, String ID);

    @Positive
    @Pure
    @Positive
    public final String getDisplayName();

    @Positive
    @Pure
    @Positive
    public final String getDisplayName(Locale locale);

    @Positive
    @Pure
    @Positive
    public final String getDisplayName(boolean daylight, int style);

    @Positive
    @Pure
    @Positive
    public String getDisplayName(boolean daylight, int style, Locale locale);

    @Positive
    @Pure
    @Positive
    public int getDSTSavings();

    @Positive
    @Pure
    @Positive
    public abstract boolean useDaylightTime();

    @Positive
    public boolean observesDaylightTime();

    @Positive
    @Pure
    @Positive
    public abstract boolean inDaylightTime(Date date);

    @Positive
    @Pure
    @Positive
    public static synchronized TimeZone getTimeZone(String ID);

    @Positive
    public static TimeZone getTimeZone(ZoneId zoneId);

    @Positive
    public ZoneId toZoneId();

    @Positive
    public static synchronized String[] getAvailableIDs(int rawOffset);

    @Positive
    public static synchronized String[] getAvailableIDs();

    @Positive
    public static TimeZone getDefault();

    @Positive
    static TimeZone getDefaultRef();

    @Positive
    public static void setDefault(@Nullable TimeZone zone);

    @Positive
    public boolean hasSameRules(@Nullable TimeZone other);

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied TimeZone this);
    @Positive
}
