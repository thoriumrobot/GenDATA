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
import static java.time.temporal.ChronoField.EPOCH_DAY;
    @Positive
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
    @Positive
import static java.time.temporal.ChronoUnit.FOREVER;
    @Positive
import static java.time.temporal.ChronoUnit.NANOS;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectOutput;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.chrono.IsoChronology;
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
import java.util.Comparator;
    @Positive
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class OffsetDateTime implements Temporal, TemporalAdjuster, Comparable<OffsetDateTime>, Serializable {

    @Positive
    public static final OffsetDateTime MIN;

    @Positive
    public static final OffsetDateTime MAX;

    @Positive
    public static Comparator<OffsetDateTime> timeLineOrder();

    @Positive
    public static OffsetDateTime now();

    @Positive
    public static OffsetDateTime now(ZoneId zone);

    @Positive
    public static OffsetDateTime now(Clock clock);

    @Positive
    public static OffsetDateTime of(LocalDate date, LocalTime time, ZoneOffset offset);

    @Positive
    public static OffsetDateTime of(LocalDateTime dateTime, ZoneOffset offset);

    @Positive
    public static OffsetDateTime of(int year, int month, int dayOfMonth, int hour, int minute, int second, int nanoOfSecond, ZoneOffset offset);

    @Positive
    public static OffsetDateTime ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static OffsetDateTime from(TemporalAccessor temporal);

    @Positive
    public static OffsetDateTime parse(CharSequence text);

    @Positive
    public static OffsetDateTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public ZoneOffset getOffset();

    @Positive
    public OffsetDateTime withOffsetSameLocal(ZoneOffset offset);

    @Positive
    public OffsetDateTime withOffsetSameInstant(ZoneOffset offset);

    @Positive
    public LocalDateTime toLocalDateTime();

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
    public OffsetDateTime with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public OffsetDateTime with(TemporalField field, long newValue);

    @Positive
    public OffsetDateTime withYear(int year);

    @Positive
    public OffsetDateTime withMonth(int month);

    @Positive
    public OffsetDateTime withDayOfMonth(int dayOfMonth);

    @Positive
    public OffsetDateTime withDayOfYear(int dayOfYear);

    @Positive
    public OffsetDateTime withHour(int hour);

    @Positive
    public OffsetDateTime withMinute(int minute);

    @Positive
    public OffsetDateTime withSecond(int second);

    @Positive
    public OffsetDateTime withNano(int nanoOfSecond);

    @Positive
    public OffsetDateTime truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public OffsetDateTime plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public OffsetDateTime plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public OffsetDateTime plusYears(long years);

    @Positive
    public OffsetDateTime plusMonths(long months);

    @Positive
    public OffsetDateTime plusWeeks(long weeks);

    @Positive
    public OffsetDateTime plusDays(long days);

    @Positive
    public OffsetDateTime plusHours(long hours);

    @Positive
    public OffsetDateTime plusMinutes(long minutes);

    @Positive
    public OffsetDateTime plusSeconds(long seconds);

    @Positive
    public OffsetDateTime plusNanos(long nanos);

    @Positive
    @Override
    @Positive
    public OffsetDateTime minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public OffsetDateTime minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public OffsetDateTime minusYears(long years);

    @Positive
    public OffsetDateTime minusMonths(long months);

    @Positive
    public OffsetDateTime minusWeeks(long weeks);

    @Positive
    public OffsetDateTime minusDays(long days);

    @Positive
    public OffsetDateTime minusHours(long hours);

    @Positive
    public OffsetDateTime minusMinutes(long minutes);

    @Positive
    public OffsetDateTime minusSeconds(long seconds);

    @Positive
    public OffsetDateTime minusNanos(long nanos);

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
    public String format(DateTimeFormatter formatter);

    @Positive
    public ZonedDateTime atZoneSameInstant(ZoneId zone);

    @Positive
    public ZonedDateTime atZoneSimilarLocal(ZoneId zone);

    @Positive
    public OffsetTime toOffsetTime();

    @Positive
    public ZonedDateTime toZonedDateTime();

    @Positive
    public Instant toInstant();

    @Positive
    public long toEpochSecond();

    @Positive
    @Override
    @Positive
    public int compareTo(OffsetDateTime other);

    @Positive
    public boolean isAfter(OffsetDateTime other);

    @Positive
    public boolean isBefore(OffsetDateTime other);

    @Positive
    public boolean isEqual(OffsetDateTime other);

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
    void writeExternal(ObjectOutput out) throws IOException;

    @Positive
    static OffsetDateTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 1
