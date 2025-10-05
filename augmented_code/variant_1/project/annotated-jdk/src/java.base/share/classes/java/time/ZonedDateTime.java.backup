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
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
import java.io.DataOutput;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.chrono.ChronoZonedDateTime;
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
import java.util.List;
import java.util.Objects;

@jdk.internal.ValueBased
public final class ZonedDateTime implements Temporal, ChronoZonedDateTime<LocalDate>, Serializable {

    public static ZonedDateTime now();

    public static ZonedDateTime now(ZoneId zone);

    public static ZonedDateTime now(Clock clock);

    public static ZonedDateTime of(LocalDate date, LocalTime time, ZoneId zone);

    public static ZonedDateTime of(LocalDateTime localDateTime, ZoneId zone);

    public static ZonedDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond, ZoneId zone);

    public static ZonedDateTime ofLocal(LocalDateTime localDateTime, ZoneId zone, ZoneOffset preferredOffset);

    public static ZonedDateTime ofInstant(Instant instant, ZoneId zone);

    public static ZonedDateTime ofInstant(LocalDateTime localDateTime, ZoneOffset offset, ZoneId zone);

    public static ZonedDateTime ofStrict(LocalDateTime localDateTime, ZoneOffset offset, ZoneId zone);

    public static ZonedDateTime from(TemporalAccessor temporal);

    public static ZonedDateTime parse(CharSequence text);

    public static ZonedDateTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public ZoneOffset getOffset();

    @Override
    public ZonedDateTime withEarlierOffsetAtOverlap();

    @Override
    public ZonedDateTime withLaterOffsetAtOverlap();

    @Override
    public ZoneId getZone();

    @Override
    public ZonedDateTime withZoneSameLocal(ZoneId zone);

    @Override
    public ZonedDateTime withZoneSameInstant(ZoneId zone);

    public ZonedDateTime withFixedOffsetZone();

    @Override
    public LocalDateTime toLocalDateTime();

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
    public ZonedDateTime with(TemporalAdjuster adjuster);

    @Override
    public ZonedDateTime with(TemporalField field, long newValue);

    public ZonedDateTime withYear(int year);

    public ZonedDateTime withMonth(int month);

    public ZonedDateTime withDayOfMonth(int dayOfMonth);

    public ZonedDateTime withDayOfYear(int dayOfYear);

    public ZonedDateTime withHour(int hour);

    public ZonedDateTime withMinute(int minute);

    public ZonedDateTime withSecond(int second);

    public ZonedDateTime withNano(int nanoOfSecond);

    public ZonedDateTime truncatedTo(TemporalUnit unit);

    @Override
    public ZonedDateTime plus(TemporalAmount amountToAdd);

    @Override
    public ZonedDateTime plus(long amountToAdd, TemporalUnit unit);

    public ZonedDateTime plusYears(long years);

    public ZonedDateTime plusMonths(long months);

    public ZonedDateTime plusWeeks(long weeks);

    public ZonedDateTime plusDays(long days);

    public ZonedDateTime plusHours(long hours);

    public ZonedDateTime plusMinutes(long minutes);

    public ZonedDateTime plusSeconds(long seconds);

    public ZonedDateTime plusNanos(long nanos);

    @Override
    public ZonedDateTime minus(TemporalAmount amountToSubtract);

    @Override
    public ZonedDateTime minus(long amountToSubtract, TemporalUnit unit);

    public ZonedDateTime minusYears(long years);

    public ZonedDateTime minusMonths(long months);

    public ZonedDateTime minusWeeks(long weeks);

    public ZonedDateTime minusDays(long days);

    public ZonedDateTime minusHours(long hours);

    public ZonedDateTime minusMinutes(long minutes);

    public ZonedDateTime minusSeconds(long seconds);

    public ZonedDateTime minusNanos(long nanos);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Override
    public String format(DateTimeFormatter formatter);

    public OffsetDateTime toOffsetDateTime();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static ZonedDateTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
}
