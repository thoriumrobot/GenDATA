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
package java.time.chrono;

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
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_MONTH;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_YEAR;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_MONTH;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
    @Positive
import static java.time.temporal.ChronoField.DAY_OF_WEEK;
    @Positive
import static java.time.temporal.ChronoField.DAY_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.EPOCH_DAY;
    @Positive
import static java.time.temporal.ChronoField.ERA;
    @Positive
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
    @Positive
import static java.time.temporal.ChronoField.YEAR;
    @Positive
import static java.time.temporal.ChronoField.YEAR_OF_ERA;
    @Positive
import static java.time.temporal.ChronoUnit.DAYS;
    @Positive
import static java.time.temporal.ChronoUnit.MONTHS;
    @Positive
import static java.time.temporal.ChronoUnit.WEEKS;
    @Positive
import static java.time.temporal.TemporalAdjusters.nextOrSame;
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
import java.io.ObjectStreamException;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.DateTimeException;
    @Positive
import java.time.DayOfWeek;
    @Positive
import java.time.format.ResolverStyle;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.TemporalAdjusters;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
public abstract class AbstractChronology implements Chronology {

    @Positive
    static Chronology registerChrono(Chronology chrono);

    @Positive
    static Chronology registerChrono(Chronology chrono, String id);

    @Positive
    static Chronology ofLocale(Locale locale);

    @Positive
    static Chronology of(String id);

    @Positive
    static Set<Chronology> getAvailableChronologies();

    @Positive
    protected AbstractChronology() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public ChronoLocalDate resolveDate(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    void resolveProlepticMonth(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYearOfEra(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYMD(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYD(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYMAA(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYMAD(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYAA(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveYAD(Map<TemporalField, Long> fieldValues, ResolverStyle resolverStyle);

    @Positive
    ChronoLocalDate resolveAligned(ChronoLocalDate base, long months, long weeks, long dow);

    @Positive
    void addFieldValue(Map<TemporalField, Long> fieldValues, ChronoField field, long value);

    @Positive
    @Override
    @Positive
    public int compareTo(Chronology other);

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
    @java.io.Serial
    @Positive
    Object writeReplace();

    @Positive
    void writeExternal(DataOutput out) throws IOException;

    @Positive
    static Chronology readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
