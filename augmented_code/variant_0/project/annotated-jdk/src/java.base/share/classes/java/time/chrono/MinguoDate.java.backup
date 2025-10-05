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
import static java.time.chrono.MinguoChronology.YEARS_DIFFERENCE;
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.YEAR;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.Clock;
import java.time.DateTimeException;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.Period;
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
import java.util.Objects;

@jdk.internal.ValueBased
public final class MinguoDate extends ChronoLocalDateImpl<MinguoDate> implements ChronoLocalDate, Serializable {

    public static MinguoDate now();

    public static MinguoDate now(ZoneId zone);

    public static MinguoDate now(Clock clock);

    public static MinguoDate of(int prolepticYear, int month, int dayOfMonth);

    public static MinguoDate from(TemporalAccessor temporal);

    @Override
    public MinguoChronology getChronology();

    @Override
    public MinguoEra getEra();

    @Override
    public int lengthOfMonth();

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @Override
    public MinguoDate with(TemporalField field, long newValue);

    @Override
    public MinguoDate with(TemporalAdjuster adjuster);

    @Override
    public MinguoDate plus(TemporalAmount amount);

    @Override
    public MinguoDate minus(TemporalAmount amount);

    @Override
    MinguoDate plusYears(long years);

    @Override
    MinguoDate plusMonths(long months);

    @Override
    MinguoDate plusWeeks(long weeksToAdd);

    @Override
    MinguoDate plusDays(long days);

    @Override
    public MinguoDate plus(long amountToAdd, TemporalUnit unit);

    @Override
    public MinguoDate minus(long amountToAdd, TemporalUnit unit);

    @Override
    MinguoDate minusYears(long yearsToSubtract);

    @Override
    MinguoDate minusMonths(long monthsToSubtract);

    @Override
    MinguoDate minusWeeks(long weeksToSubtract);

    @Override
    MinguoDate minusDays(long daysToSubtract);

    @Override
    @SuppressWarnings("unchecked")
    public final ChronoLocalDateTime<MinguoDate> atTime(LocalTime localTime);

    @Override
    public ChronoPeriod until(ChronoLocalDate endDate);

    @Override
    public long toEpochDay();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    void writeExternal(DataOutput out) throws IOException;

    static MinguoDate readExternal(DataInput in) throws IOException;
}
