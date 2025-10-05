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
package java.time;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.LocalTime.SECONDS_PER_DAY;
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_MONTH;
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_YEAR;
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_MONTH;
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_YEAR;
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
import static java.time.temporal.ChronoField.DAY_OF_YEAR;
import static java.time.temporal.ChronoField.EPOCH_DAY;
import static java.time.temporal.ChronoField.ERA;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
import static java.time.temporal.ChronoField.YEAR;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.chrono.ChronoLocalDate;
import java.time.chrono.IsoEra;
import java.time.chrono.IsoChronology;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAccessor;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalField;
import java.time.temporal.TemporalQueries;
import java.time.temporal.TemporalQuery;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.time.temporal.ValueRange;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;
import java.util.Objects;
import java.util.stream.LongStream;
import java.util.stream.Stream;

@jdk.internal.ValueBased
public final class LocalDate implements Temporal, TemporalAdjuster, ChronoLocalDate, Serializable {

    public static final LocalDate MIN;

    public static final LocalDate MAX;

    public static final LocalDate EPOCH;

    public static LocalDate now();

    public static LocalDate now(ZoneId zone);

    public static LocalDate now(Clock clock);

    public static LocalDate of(int year, Month month, int dayOfMonth);

    public static LocalDate of(int year, int month, int dayOfMonth);

    public static LocalDate ofYearDay(int year, int dayOfYear);

    public static LocalDate ofInstant(Instant instant, ZoneId zone);

    public static LocalDate ofEpochDay(long epochDay);

    public static LocalDate from(TemporalAccessor temporal);

    public static LocalDate parse(CharSequence text);

    public static LocalDate parse(CharSequence text, DateTimeFormatter formatter);

    @Override
    public boolean isSupported(TemporalField field);

    @Override
    public boolean isSupported(TemporalUnit unit);

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public int get(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @Override
    public IsoChronology getChronology();

    @Override
    public IsoEra getEra();

    public int getYear();

    public int getMonthValue();

    public Month getMonth();

    public int getDayOfMonth();

    public int getDayOfYear();

    public DayOfWeek getDayOfWeek();

    @Override
    public boolean isLeapYear();

    @Override
    public int lengthOfMonth();

    @Override
    public int lengthOfYear();

    @Override
    public LocalDate with(TemporalAdjuster adjuster);

    @Override
    public LocalDate with(TemporalField field, long newValue);

    public LocalDate withYear(int year);

    public LocalDate withMonth(int month);

    public LocalDate withDayOfMonth(int dayOfMonth);

    public LocalDate withDayOfYear(int dayOfYear);

    @Override
    public LocalDate plus(TemporalAmount amountToAdd);

    @Override
    public LocalDate plus(long amountToAdd, TemporalUnit unit);

    public LocalDate plusYears(long yearsToAdd);

    public LocalDate plusMonths(long monthsToAdd);

    public LocalDate plusWeeks(long weeksToAdd);

    public LocalDate plusDays(long daysToAdd);

    @Override
    public LocalDate minus(TemporalAmount amountToSubtract);

    @Override
    public LocalDate minus(long amountToSubtract, TemporalUnit unit);

    public LocalDate minusYears(long yearsToSubtract);

    public LocalDate minusMonths(long monthsToSubtract);

    public LocalDate minusWeeks(long weeksToSubtract);

    public LocalDate minusDays(long daysToSubtract);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    long daysUntil(LocalDate end);

    @Override
    public Period until(ChronoLocalDate endDateExclusive);

    public Stream<LocalDate> datesUntil(LocalDate endExclusive);

    public Stream<LocalDate> datesUntil(LocalDate endExclusive, Period step);

    @Override
    public String format(DateTimeFormatter formatter);

    @Override
    public LocalDateTime atTime(LocalTime time);

    public LocalDateTime atTime(int hour, int minute);

    public LocalDateTime atTime(int hour, int minute, int second);

    public LocalDateTime atTime(int hour, int minute, int second, int nanoOfSecond);

    public OffsetDateTime atTime(OffsetTime time);

    public LocalDateTime atStartOfDay();

    public ZonedDateTime atStartOfDay(ZoneId zone);

    @Override
    public long toEpochDay();

    public long toEpochSecond(LocalTime time, ZoneOffset offset);

    @Override
    public int compareTo(ChronoLocalDate other);

    int compareTo0(LocalDate otherDate);

    @Override
    public boolean isAfter(ChronoLocalDate other);

    @Override
    public boolean isBefore(ChronoLocalDate other);

    @Override
    public boolean isEqual(ChronoLocalDate other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static LocalDate readExternal(DataInput in) throws IOException;
}
