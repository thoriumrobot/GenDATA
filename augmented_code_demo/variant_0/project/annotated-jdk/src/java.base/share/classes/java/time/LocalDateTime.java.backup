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
import static java.time.LocalTime.HOURS_PER_DAY;
import static java.time.LocalTime.MICROS_PER_DAY;
import static java.time.LocalTime.MILLIS_PER_DAY;
import static java.time.LocalTime.MINUTES_PER_DAY;
import static java.time.LocalTime.NANOS_PER_DAY;
import static java.time.LocalTime.NANOS_PER_HOUR;
import static java.time.LocalTime.NANOS_PER_MINUTE;
import static java.time.LocalTime.NANOS_PER_SECOND;
import static java.time.LocalTime.SECONDS_PER_DAY;
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.chrono.ChronoLocalDateTime;
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
import java.time.zone.ZoneRules;
import java.util.Objects;

@jdk.internal.ValueBased
public final class LocalDateTime implements Temporal, TemporalAdjuster, ChronoLocalDateTime<LocalDate>, Serializable {

    public static final LocalDateTime MIN;

    public static final LocalDateTime MAX;

    public static LocalDateTime now();

    public static LocalDateTime now(ZoneId zone);

    public static LocalDateTime now(Clock clock);

    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute);

    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute, int second);

    public static LocalDateTime of(int year, Month month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond);

    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute);

    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second);

    public static LocalDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond);

    public static LocalDateTime of(LocalDate date, LocalTime time);

    public static LocalDateTime ofInstant(Instant instant, ZoneId zone);

    public static LocalDateTime ofEpochSecond(long epochSecond, int nanoOfSecond, ZoneOffset offset);

    public static LocalDateTime from(TemporalAccessor temporal);

    public static LocalDateTime parse(CharSequence text);

    public static LocalDateTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public LocalDate toLocalDate();

    public int getYear();

    public int getMonthValue();

    public Month getMonth();

    public int getDayOfMonth();

    public int getDayOfYear();

    public DayOfWeek getDayOfWeek();

    @Override
    public LocalTime toLocalTime();

    public int getHour();

    public int getMinute();

    public int getSecond();

    public int getNano();

    @Override
    public LocalDateTime with(TemporalAdjuster adjuster);

    @Override
    public LocalDateTime with(TemporalField field, long newValue);

    public LocalDateTime withYear(int year);

    public LocalDateTime withMonth(int month);

    public LocalDateTime withDayOfMonth(int dayOfMonth);

    public LocalDateTime withDayOfYear(int dayOfYear);

    public LocalDateTime withHour(int hour);

    public LocalDateTime withMinute(int minute);

    public LocalDateTime withSecond(int second);

    public LocalDateTime withNano(int nanoOfSecond);

    public LocalDateTime truncatedTo(TemporalUnit unit);

    @Override
    public LocalDateTime plus(TemporalAmount amountToAdd);

    @Override
    public LocalDateTime plus(long amountToAdd, TemporalUnit unit);

    public LocalDateTime plusYears(long years);

    public LocalDateTime plusMonths(long months);

    public LocalDateTime plusWeeks(long weeks);

    public LocalDateTime plusDays(long days);

    public LocalDateTime plusHours(long hours);

    public LocalDateTime plusMinutes(long minutes);

    public LocalDateTime plusSeconds(long seconds);

    public LocalDateTime plusNanos(long nanos);

    @Override
    public LocalDateTime minus(TemporalAmount amountToSubtract);

    @Override
    public LocalDateTime minus(long amountToSubtract, TemporalUnit unit);

    public LocalDateTime minusYears(long years);

    public LocalDateTime minusMonths(long months);

    public LocalDateTime minusWeeks(long weeks);

    public LocalDateTime minusDays(long days);

    public LocalDateTime minusHours(long hours);

    public LocalDateTime minusMinutes(long minutes);

    public LocalDateTime minusSeconds(long seconds);

    public LocalDateTime minusNanos(long nanos);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Override
    public String format(DateTimeFormatter formatter);

    public OffsetDateTime atOffset(ZoneOffset offset);

    @Override
    public ZonedDateTime atZone(ZoneId zone);

    @Override
    public int compareTo(ChronoLocalDateTime<?> other);

    @Override
    public boolean isAfter(ChronoLocalDateTime<?> other);

    @Override
    public boolean isBefore(ChronoLocalDateTime<?> other);

    @Override
    public boolean isEqual(ChronoLocalDateTime<?> other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static LocalDateTime readExternal(DataInput in) throws IOException;
}
