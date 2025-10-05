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
import static java.time.chrono.ThaiBuddhistChronology.YEARS_DIFFERENCE;
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
public final class ThaiBuddhistDate extends ChronoLocalDateImpl<ThaiBuddhistDate> implements ChronoLocalDate, Serializable {

    public static ThaiBuddhistDate now();

    public static ThaiBuddhistDate now(ZoneId zone);

    public static ThaiBuddhistDate now(Clock clock);

    public static ThaiBuddhistDate of(int prolepticYear, int month, int dayOfMonth);

    public static ThaiBuddhistDate from(TemporalAccessor temporal);

    @Override
    public ThaiBuddhistChronology getChronology();

    @Override
    public ThaiBuddhistEra getEra();

    @Override
    public int lengthOfMonth();

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @Override
    public ThaiBuddhistDate with(TemporalField field, long newValue);

    @Override
    public ThaiBuddhistDate with(TemporalAdjuster adjuster);

    @Override
    public ThaiBuddhistDate plus(TemporalAmount amount);

    @Override
    public ThaiBuddhistDate minus(TemporalAmount amount);

    @Override
    ThaiBuddhistDate plusYears(long years);

    @Override
    ThaiBuddhistDate plusMonths(long months);

    @Override
    ThaiBuddhistDate plusWeeks(long weeksToAdd);

    @Override
    ThaiBuddhistDate plusDays(long days);

    @Override
    public ThaiBuddhistDate plus(long amountToAdd, TemporalUnit unit);

    @Override
    public ThaiBuddhistDate minus(long amountToAdd, TemporalUnit unit);

    @Override
    ThaiBuddhistDate minusYears(long yearsToSubtract);

    @Override
    ThaiBuddhistDate minusMonths(long monthsToSubtract);

    @Override
    ThaiBuddhistDate minusWeeks(long weeksToSubtract);

    @Override
    ThaiBuddhistDate minusDays(long daysToSubtract);

    @Override
    @SuppressWarnings("unchecked")
    public final ChronoLocalDateTime<ThaiBuddhistDate> atTime(LocalTime localTime);

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

    static ThaiBuddhistDate readExternal(DataInput in) throws IOException;
}
