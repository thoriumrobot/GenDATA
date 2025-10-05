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
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
    @Positive
import java.io.DataOutput;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.chrono.ChronoZonedDateTime;
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
import java.time.zone.ZoneOffsetTransition;
    @Positive
import java.time.zone.ZoneRules;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class ZonedDateTime implements Temporal, ChronoZonedDateTime<LocalDate>, Serializable {

    @Positive
    public static ZonedDateTime now();

    @Positive
    public static ZonedDateTime now(ZoneId zone);

    @Positive
    public static ZonedDateTime now(Clock clock);

    @Positive
    public static ZonedDateTime of(LocalDate date, LocalTime time, ZoneId zone);

    @Positive
    public static ZonedDateTime of(LocalDateTime localDateTime, ZoneId zone);

    @Positive
    public static ZonedDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond, ZoneId zone);

    @Positive
    public static ZonedDateTime ofLocal(LocalDateTime localDateTime, ZoneId zone, ZoneOffset preferredOffset);

    @Positive
    public static ZonedDateTime ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static ZonedDateTime ofInstant(LocalDateTime localDateTime, ZoneOffset offset, ZoneId zone);

    @Positive
    public static ZonedDateTime ofStrict(LocalDateTime localDateTime, ZoneOffset offset, ZoneId zone);

    @Positive
    public static ZonedDateTime from(TemporalAccessor temporal);

    @Positive
    public static ZonedDateTime parse(CharSequence text);

    @Positive
    public static ZonedDateTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public ZoneOffset getOffset();

    @Positive
    @Override
    @Positive
    public ZonedDateTime withEarlierOffsetAtOverlap();

    @Positive
    @Override
    @Positive
    public ZonedDateTime withLaterOffsetAtOverlap();

    @Positive
    @Override
    @Positive
    public ZoneId getZone();

    @Positive
    @Override
    @Positive
    public ZonedDateTime withZoneSameLocal(ZoneId zone);

    @Positive
    @Override
    @Positive
    public ZonedDateTime withZoneSameInstant(ZoneId zone);

    @Positive
    public ZonedDateTime withFixedOffsetZone();

    @Positive
    @Override
    @Positive
    public LocalDateTime toLocalDateTime();

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
    public ZonedDateTime with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public ZonedDateTime with(TemporalField field, long newValue);

    @Positive
    public ZonedDateTime withYear(int year);

    @Positive
    public ZonedDateTime withMonth(int month);

    @Positive
    public ZonedDateTime withDayOfMonth(int dayOfMonth);

    @Positive
    public ZonedDateTime withDayOfYear(int dayOfYear);

    @Positive
    public ZonedDateTime withHour(int hour);

    @Positive
    public ZonedDateTime withMinute(int minute);

    @Positive
    public ZonedDateTime withSecond(int second);

    @Positive
    public ZonedDateTime withNano(int nanoOfSecond);

    @Positive
    public ZonedDateTime truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public ZonedDateTime plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public ZonedDateTime plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public ZonedDateTime plusYears(long years);

    @Positive
    public ZonedDateTime plusMonths(long months);

    @Positive
    public ZonedDateTime plusWeeks(long weeks);

    @Positive
    public ZonedDateTime plusDays(long days);

    @Positive
    public ZonedDateTime plusHours(long hours);

    @Positive
    public ZonedDateTime plusMinutes(long minutes);

    @Positive
    public ZonedDateTime plusSeconds(long seconds);

    @Positive
    public ZonedDateTime plusNanos(long nanos);

    @Positive
    @Override
    @Positive
    public ZonedDateTime minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public ZonedDateTime minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public ZonedDateTime minusYears(long years);

    @Positive
    public ZonedDateTime minusMonths(long months);

    @Positive
    public ZonedDateTime minusWeeks(long weeks);

    @Positive
    public ZonedDateTime minusDays(long days);

    @Positive
    public ZonedDateTime minusHours(long hours);

    @Positive
    public ZonedDateTime minusMinutes(long minutes);

    @Positive
    public ZonedDateTime minusSeconds(long seconds);

    @Positive
    public ZonedDateTime minusNanos(long nanos);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public <R> R query(TemporalQuery<R> query);

    @Positive
    @Override
    @Positive
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public String format(DateTimeFormatter formatter);

    @Positive
    public OffsetDateTime toOffsetDateTime();

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
    static ZonedDateTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 0
