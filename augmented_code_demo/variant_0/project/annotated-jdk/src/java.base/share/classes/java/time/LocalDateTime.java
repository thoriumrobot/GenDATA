/*
    @Positive
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time;

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
import static java.time.LocalTime.HOURS_PER_DAY;
    @Positive
import static java.time.LocalTime.MICROS_PER_DAY;
    @Positive
import static java.time.LocalTime.MILLIS_PER_DAY;
    @Positive
import static java.time.LocalTime.MINUTES_PER_DAY;
    @Positive
import static java.time.LocalTime.NANOS_PER_DAY;
    @Positive
import static java.time.LocalTime.NANOS_PER_HOUR;
    @Positive
import static java.time.LocalTime.NANOS_PER_MINUTE;
    @Positive
import static java.time.LocalTime.NANOS_PER_SECOND;
    @Positive
import static java.time.LocalTime.SECONDS_PER_DAY;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
    @Positive
import java.io.DataInput;
    @Positive
import java.io.DataOutput;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.chrono.ChronoLocalDateTime;
    @Positive
import java.time.format.DateTimeFormatter;
    @Positive
import java.time.format.DateTimeParseException;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalAdjuster;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalQueries;
    @Positive
import java.time.temporal.TemporalQuery;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.time.zone.ZoneRules;
    @Positive
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class LocalDateTime implements Temporal, TemporalAdjuster, ChronoLocalDateTime<LocalDate>, Serializable {

    @Positive
    public static final LocalDateTime MIN;

    @Positive
    public static final LocalDateTime MAX;

    @Positive
    public static LocalDateTime now();

    @Positive
    public static LocalDateTime now(ZoneId zone);

    @Positive
    public static LocalDateTime now(Clock clock);

    @Positive
    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute);

    @Positive
    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute, int second);

    @Positive
    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond);

    @Positive
    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute);

    @Positive
    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second);

    @Positive
    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond);

    @Positive
    public static LocalDateTime of(LocalDate date, LocalTime time);

    @Positive
    public static LocalDateTime ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static LocalDateTime ofEpochSecond(long epochSecond, int nanoOfSecond, ZoneOffset offset);

    @Positive
    public static LocalDateTime from(TemporalAccessor temporal);

    @Positive
    public static LocalDateTime parse(CharSequence text);

    @Positive
    public static LocalDateTime parse(CharSequence text, DateTimeFormatter formatter);

    @Positive
    @Override
    @Positive
    public boolean isSupported(TemporalField field);

    @Positive
    @Override
    @Positive
    public boolean isSupported(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public ValueRange range(TemporalField field);

    @Positive
    @Override
    @Positive
    public int get(TemporalField field);

    @Positive
    @Override
    @Positive
    public long getLong(TemporalField field);

    @Positive
    @Override
    @Positive
    public LocalDate toLocalDate();

    @Positive
    public int getYear();

    @Positive
    public int getMonthValue();

    @Positive
    public Month getMonth();

    @Positive
    public int getDayOfMonth();

    @Positive
    public int getDayOfYear();

    @Positive
    public DayOfWeek getDayOfWeek();

    @Positive
    @Override
    @Positive
    public LocalTime toLocalTime();

    @Positive
    public int getHour();

    @Positive
    public int getMinute();

    @Positive
    public int getSecond();

    @Positive
    public int getNano();

    @Positive
    @Override
    @Positive
    public LocalDateTime with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public LocalDateTime with(TemporalField field, long newValue);

    @Positive
    public LocalDateTime withYear(int year);

    @Positive
    public LocalDateTime withMonth(int month);

    @Positive
    public LocalDateTime withDayOfMonth(int dayOfMonth);

    @Positive
    public LocalDateTime withDayOfYear(int dayOfYear);

    @Positive
    public LocalDateTime withHour(int hour);

    @Positive
    public LocalDateTime withMinute(int minute);

    @Positive
    public LocalDateTime withSecond(int second);

    @Positive
    public LocalDateTime withNano(int nanoOfSecond);

    @Positive
    public LocalDateTime truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public LocalDateTime plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public LocalDateTime plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public LocalDateTime plusYears(long years);

    @Positive
    public LocalDateTime plusMonths(long months);

    @Positive
    public LocalDateTime plusWeeks(long weeks);

    @Positive
    public LocalDateTime plusDays(long days);

    @Positive
    public LocalDateTime plusHours(long hours);

    @Positive
    public LocalDateTime plusMinutes(long minutes);

    @Positive
    public LocalDateTime plusSeconds(long seconds);

    @Positive
    public LocalDateTime plusNanos(long nanos);

    @Positive
    @Override
    @Positive
    public LocalDateTime minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public LocalDateTime minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public LocalDateTime minusYears(long years);

    @Positive
    public LocalDateTime minusMonths(long months);

    @Positive
    public LocalDateTime minusWeeks(long weeks);

    @Positive
    public LocalDateTime minusDays(long days);

    @Positive
    public LocalDateTime minusHours(long hours);

    @Positive
    public LocalDateTime minusMinutes(long minutes);

    @Positive
    public LocalDateTime minusSeconds(long seconds);

    @Positive
    public LocalDateTime minusNanos(long nanos);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public <R> R query(TemporalQuery<R> query);

    @Positive
    @Override
    @Positive
    public Temporal adjustInto(Temporal temporal);

    @Positive
    @Override
    @Positive
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public String format(DateTimeFormatter formatter);

    @Positive
    public OffsetDateTime atOffset(ZoneOffset offset);

    @Positive
    @Override
    @Positive
    public ZonedDateTime atZone(ZoneId zone);

    @Positive
    @Override
    @Positive
    public int compareTo(ChronoLocalDateTime<?> other);

    @Positive
    @Override
    @Positive
    public boolean isAfter(ChronoLocalDateTime<?> other);

    @Positive
    @Override
    @Positive
    public boolean isBefore(ChronoLocalDateTime<?> other);

    @Positive
    @Override
    @Positive
    public boolean isEqual(ChronoLocalDateTime<?> other);

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
    public String toString();

    @Positive
    void writeExternal(DataOutput out) throws IOException;

    @Positive
    static LocalDateTime readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
