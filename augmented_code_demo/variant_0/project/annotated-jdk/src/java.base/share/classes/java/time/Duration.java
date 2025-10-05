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
import static java.time.LocalTime.MINUTES_PER_HOUR;
    @Positive
import static java.time.LocalTime.NANOS_PER_MILLI;
    @Positive
import static java.time.LocalTime.NANOS_PER_SECOND;
    @Positive
import static java.time.LocalTime.SECONDS_PER_DAY;
    @Positive
import static java.time.LocalTime.SECONDS_PER_HOUR;
    @Positive
import static java.time.LocalTime.SECONDS_PER_MINUTE;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoUnit.DAYS;
    @Positive
import static java.time.temporal.ChronoUnit.NANOS;
    @Positive
import static java.time.temporal.ChronoUnit.SECONDS;
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
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.RoundingMode;
    @Positive
import java.time.format.DateTimeParseException;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Duration implements TemporalAmount, Comparable<Duration>, Serializable {

    @Positive
    public static final Duration ZERO;

    @Positive
    private static class Lazy {
    @Positive
    }

    @Positive
    public static Duration ofDays(long days);

    @Positive
    public static Duration ofHours(long hours);

    @Positive
    public static Duration ofMinutes(long minutes);

    @Positive
    public static Duration ofSeconds(long seconds);

    @Positive
    public static Duration ofSeconds(long seconds, long nanoAdjustment);

    @Positive
    public static Duration ofMillis(long millis);

    @Positive
    public static Duration ofNanos(long nanos);

    @Positive
    public static Duration of(long amount, TemporalUnit unit);

    @Positive
    public static Duration from(TemporalAmount amount);

    @Positive
    public static Duration parse(CharSequence text);

    @Positive
    public static Duration between(Temporal startInclusive, Temporal endExclusive);

    @Positive
    @Override
    @Positive
    public long get(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public List<TemporalUnit> getUnits();

    @Positive
    private static class DurationUnits {
    @Positive
    }

    @Positive
    public boolean isZero();

    @Positive
    public boolean isNegative();

    @Positive
    public long getSeconds();

    @Positive
    public int getNano();

    @Positive
    public Duration withSeconds(long seconds);

    @Positive
    public Duration withNanos(int nanoOfSecond);

    @Positive
    public Duration plus(Duration duration);

    @Positive
    public Duration plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public Duration plusDays(long daysToAdd);

    @Positive
    public Duration plusHours(long hoursToAdd);

    @Positive
    public Duration plusMinutes(long minutesToAdd);

    @Positive
    public Duration plusSeconds(long secondsToAdd);

    @Positive
    public Duration plusMillis(long millisToAdd);

    @Positive
    public Duration plusNanos(long nanosToAdd);

    @Positive
    public Duration minus(Duration duration);

    @Positive
    public Duration minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public Duration minusDays(long daysToSubtract);

    @Positive
    public Duration minusHours(long hoursToSubtract);

    @Positive
    public Duration minusMinutes(long minutesToSubtract);

    @Positive
    public Duration minusSeconds(long secondsToSubtract);

    @Positive
    public Duration minusMillis(long millisToSubtract);

    @Positive
    public Duration minusNanos(long nanosToSubtract);

    @Positive
    public Duration multipliedBy(long multiplicand);

    @Positive
    public Duration dividedBy(long divisor);

    @Positive
    public long dividedBy(Duration divisor);

    @Positive
    public Duration negated();

    @Positive
    public Duration abs();

    @Positive
    @Override
    @Positive
    public Temporal addTo(Temporal temporal);

    @Positive
    @Override
    @Positive
    public Temporal subtractFrom(Temporal temporal);

    @Positive
    public long toDays();

    @Positive
    public long toHours();

    @Positive
    public long toMinutes();

    @Positive
    public long toSeconds();

    @Positive
    public long toMillis();

    @Positive
    public long toNanos();

    @Positive
    public long toDaysPart();

    @Positive
    public int toHoursPart();

    @Positive
    public int toMinutesPart();

    @Positive
    public int toSecondsPart();

    @Positive
    public int toMillisPart();

    @Positive
    public int toNanosPart();

    @Positive
    public Duration truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public int compareTo(Duration otherDuration);

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
    static Duration readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
