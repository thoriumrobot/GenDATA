/*
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time.chrono;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_MONTH;
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_YEAR;
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_MONTH;
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_YEAR;
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.YEAR;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.time.Clock;
import java.time.DateTimeException;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.temporal.ChronoField;
import java.time.temporal.TemporalAccessor;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalField;
import java.time.temporal.TemporalQuery;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.time.temporal.ValueRange;

@jdk.internal.ValueBased
public final class HijrahDate extends ChronoLocalDateImpl<HijrahDate> implements ChronoLocalDate, Serializable {

    static HijrahDate of(HijrahChronology chrono, int prolepticYear, int monthOfYear, int dayOfMonth);

    static HijrahDate ofEpochDay(HijrahChronology chrono, long epochDay);

    public static HijrahDate now();

    public static HijrahDate now(ZoneId zone);

    public static HijrahDate now(Clock clock);

    public static HijrahDate of(int prolepticYear, int month, int dayOfMonth);

    public static HijrahDate from(TemporalAccessor temporal);

    @Override
    public HijrahChronology getChronology();

    @Override
    public HijrahEra getEra();

    @Override
    public int lengthOfMonth();

    @Override
    public int lengthOfYear();

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @Override
    public HijrahDate with(TemporalField field, long newValue);

    @Override
    public HijrahDate with(TemporalAdjuster adjuster);

    public HijrahDate withVariant(HijrahChronology chronology);

    @Override
    public HijrahDate plus(TemporalAmount amount);

    @Override
    public HijrahDate minus(TemporalAmount amount);

    @Override
    public long toEpochDay();

    @Override
    public boolean isLeapYear();

    @Override
    HijrahDate plusYears(long years);

    @Override
    HijrahDate plusMonths(long monthsToAdd);

    @Override
    HijrahDate plusWeeks(long weeksToAdd);

    @Override
    HijrahDate plusDays(long days);

    @Override
    public HijrahDate plus(long amountToAdd, TemporalUnit unit);

    @Override
    public HijrahDate minus(long amountToSubtract, TemporalUnit unit);

    @Override
    HijrahDate minusYears(long yearsToSubtract);

    @Override
    HijrahDate minusMonths(long monthsToSubtract);

    @Override
    HijrahDate minusWeeks(long weeksToSubtract);

    @Override
    HijrahDate minusDays(long daysToSubtract);

    @Override
    @SuppressWarnings("unchecked")
    public final ChronoLocalDateTime<HijrahDate> atTime(LocalTime localTime);

    @Override
    public ChronoPeriod until(ChronoLocalDate endDate);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    void writeExternal(ObjectOutput out) throws IOException;

    static HijrahDate readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
}
