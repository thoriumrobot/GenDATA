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
import java.util.Calendar;
import java.util.Objects;
import sun.util.calendar.CalendarDate;
import sun.util.calendar.LocalGregorianCalendar;

@jdk.internal.ValueBased
public final class JapaneseDate extends ChronoLocalDateImpl<JapaneseDate> implements ChronoLocalDate, Serializable {

    public static JapaneseDate now();

    public static JapaneseDate now(ZoneId zone);

    public static JapaneseDate now(Clock clock);

    public static JapaneseDate of(JapaneseEra era, int yearOfEra, int month, int dayOfMonth);

    public static JapaneseDate of(int prolepticYear, int month, int dayOfMonth);

    static JapaneseDate ofYearDay(JapaneseEra era, int yearOfEra, int dayOfYear);

    public static JapaneseDate from(TemporalAccessor temporal);

    @Override
    public JapaneseChronology getChronology();

    @Override
    public JapaneseEra getEra();

    @Override
    public int lengthOfMonth();

    @Override
    public int lengthOfYear();

    @Override
    public boolean isSupported(TemporalField field);

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @Override
    public JapaneseDate with(TemporalField field, long newValue);

    @Override
    public JapaneseDate with(TemporalAdjuster adjuster);

    @Override
    public JapaneseDate plus(TemporalAmount amount);

    @Override
    public JapaneseDate minus(TemporalAmount amount);

    @Override
    JapaneseDate plusYears(long years);

    @Override
    JapaneseDate plusMonths(long months);

    @Override
    JapaneseDate plusWeeks(long weeksToAdd);

    @Override
    JapaneseDate plusDays(long days);

    @Override
    public JapaneseDate plus(long amountToAdd, TemporalUnit unit);

    @Override
    public JapaneseDate minus(long amountToAdd, TemporalUnit unit);

    @Override
    JapaneseDate minusYears(long yearsToSubtract);

    @Override
    JapaneseDate minusMonths(long monthsToSubtract);

    @Override
    JapaneseDate minusWeeks(long weeksToSubtract);

    @Override
    JapaneseDate minusDays(long daysToSubtract);

    @Override
    @SuppressWarnings("unchecked")
    public final ChronoLocalDateTime<JapaneseDate> atTime(LocalTime localTime);

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

    static JapaneseDate readExternal(DataInput in) throws IOException;
}
