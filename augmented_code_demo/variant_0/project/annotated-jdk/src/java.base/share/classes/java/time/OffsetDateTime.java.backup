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
import static java.time.temporal.ChronoField.EPOCH_DAY;
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
import static java.time.temporal.ChronoField.NANO_OF_DAY;
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
import static java.time.temporal.ChronoUnit.FOREVER;
import static java.time.temporal.ChronoUnit.NANOS;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
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
import java.time.zone.ZoneRules;
import java.util.Comparator;
import java.util.Objects;

@jdk.internal.ValueBased
public final class OffsetDateTime implements Temporal, TemporalAdjuster, Comparable<OffsetDateTime>, Serializable {

    public static final OffsetDateTime MIN;

    public static final OffsetDateTime MAX;

    public static Comparator<OffsetDateTime> timeLineOrder();

    public static OffsetDateTime now();

    public static OffsetDateTime now(ZoneId zone);

    public static OffsetDateTime now(Clock clock);

    public static OffsetDateTime of(LocalDate date, LocalTime time, ZoneOffset offset);

    public static OffsetDateTime of(LocalDateTime dateTime, ZoneOffset offset);

    public static OffsetDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond, ZoneOffset offset);

    public static OffsetDateTime ofInstant(Instant instant, ZoneId zone);

    public static OffsetDateTime from(TemporalAccessor temporal);

    public static OffsetDateTime parse(CharSequence text);

    public static OffsetDateTime parse(CharSequence text, DateTimeFormatter formatter);

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

    public ZoneOffset getOffset();

    public OffsetDateTime withOffsetSameLocal(ZoneOffset offset);

    public OffsetDateTime withOffsetSameInstant(ZoneOffset offset);

    public LocalDateTime toLocalDateTime();

    public LocalDate toLocalDate();

    public int getYear();

    public int getMonthValue();

    public Month getMonth();

    public int getDayOfMonth();

    public int getDayOfYear();

    public DayOfWeek getDayOfWeek();

    public LocalTime toLocalTime();

    public int getHour();

    public int getMinute();

    public int getSecond();

    public int getNano();

    @Override
    public OffsetDateTime with(TemporalAdjuster adjuster);

    @Override
    public OffsetDateTime with(TemporalField field, long newValue);

    public OffsetDateTime withYear(int year);

    public OffsetDateTime withMonth(int month);

    public OffsetDateTime withDayOfMonth(int dayOfMonth);

    public OffsetDateTime withDayOfYear(int dayOfYear);

    public OffsetDateTime withHour(int hour);

    public OffsetDateTime withMinute(int minute);

    public OffsetDateTime withSecond(int second);

    public OffsetDateTime withNano(int nanoOfSecond);

    public OffsetDateTime truncatedTo(TemporalUnit unit);

    @Override
    public OffsetDateTime plus(TemporalAmount amountToAdd);

    @Override
    public OffsetDateTime plus(long amountToAdd, TemporalUnit unit);

    public OffsetDateTime plusYears(long years);

    public OffsetDateTime plusMonths(long months);

    public OffsetDateTime plusWeeks(long weeks);

    public OffsetDateTime plusDays(long days);

    public OffsetDateTime plusHours(long hours);

    public OffsetDateTime plusMinutes(long minutes);

    public OffsetDateTime plusSeconds(long seconds);

    public OffsetDateTime plusNanos(long nanos);

    @Override
    public OffsetDateTime minus(TemporalAmount amountToSubtract);

    @Override
    public OffsetDateTime minus(long amountToSubtract, TemporalUnit unit);

    public OffsetDateTime minusYears(long years);

    public OffsetDateTime minusMonths(long months);

    public OffsetDateTime minusWeeks(long weeks);

    public OffsetDateTime minusDays(long days);

    public OffsetDateTime minusHours(long hours);

    public OffsetDateTime minusMinutes(long minutes);

    public OffsetDateTime minusSeconds(long seconds);

    public OffsetDateTime minusNanos(long nanos);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    public String format(DateTimeFormatter formatter);

    public ZonedDateTime atZoneSameInstant(ZoneId zone);

    public ZonedDateTime atZoneSimilarLocal(ZoneId zone);

    public OffsetTime toOffsetTime();

    public ZonedDateTime toZonedDateTime();

    public Instant toInstant();

    public long toEpochSecond();

    @Override
    public int compareTo(OffsetDateTime other);

    public boolean isAfter(OffsetDateTime other);

    public boolean isBefore(OffsetDateTime other);

    public boolean isEqual(OffsetDateTime other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(ObjectOutput out) throws IOException;

    static OffsetDateTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
}
