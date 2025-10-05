/*
    @Positive
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
import static java.time.LocalTime.NANOS_PER_SECOND;
    @Positive
import static java.time.LocalTime.SECONDS_PER_DAY;
    @Positive
import static java.time.LocalTime.SECONDS_PER_HOUR;
    @Positive
import static java.time.LocalTime.SECONDS_PER_MINUTE;
    @Positive
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
    @Positive
import static java.time.temporal.ChronoField.MICRO_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoField.MILLI_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoUnit.DAYS;
    @Positive
import static java.time.temporal.ChronoUnit.NANOS;
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
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Instant implements Temporal, TemporalAdjuster, Comparable<Instant>, Serializable {

    @Positive
    public static final Instant EPOCH;

    @Positive
    public static final Instant MIN;

    @Positive
    public static final Instant MAX;

    @Positive
    public static Instant now();

    @Positive
    public static Instant now(Clock clock);

    @Positive
    public static Instant ofEpochSecond(long epochSecond);

    @Positive
    public static Instant ofEpochSecond(long epochSecond, long nanoAdjustment);

    @Positive
    public static Instant ofEpochMilli(long epochMilli);

    @Positive
    public static Instant from(TemporalAccessor temporal);

    @Positive
    public static Instant parse(final CharSequence text);

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
    public long getEpochSecond();

    @Positive
    public int getNano();

    @Positive
    @Override
    @Positive
    public Instant with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public Instant with(TemporalField field, long newValue);

    @Positive
    public Instant truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public Instant plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public Instant plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public Instant plusSeconds(long secondsToAdd);

    @Positive
    public Instant plusMillis(long millisToAdd);

    @Positive
    public Instant plusNanos(long nanosToAdd);

    @Positive
    @Override
    @Positive
    public Instant minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public Instant minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public Instant minusSeconds(long secondsToSubtract);

    @Positive
    public Instant minusMillis(long millisToSubtract);

    @Positive
    public Instant minusNanos(long nanosToSubtract);

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
    public OffsetDateTime atOffset(ZoneOffset offset);

    @Positive
    public ZonedDateTime atZone(ZoneId zone);

    @Positive
    public long toEpochMilli();

    @Positive
    @Override
    @Positive
    public int compareTo(Instant otherInstant);

    @Positive
    public boolean isAfter(Instant otherInstant);

    @Positive
    public boolean isBefore(Instant otherInstant);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

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
    static Instant readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
