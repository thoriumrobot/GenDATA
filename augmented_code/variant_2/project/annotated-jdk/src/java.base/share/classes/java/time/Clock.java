/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import static java.time.LocalTime.NANOS_PER_MINUTE;
    @Positive
import static java.time.LocalTime.NANOS_PER_SECOND;
    @Positive
import static java.time.LocalTime.NANOS_PER_MILLI;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Objects;
    @Positive
import java.util.TimeZone;
    @Positive
import jdk.internal.misc.VM;

    @Positive
public abstract class Clock implements InstantSource {

    @Positive
    public static Clock systemUTC();

    @Positive
    public static Clock systemDefaultZone();

    @Positive
    public static Clock system(ZoneId zone);

    @Positive
    public static Clock tickMillis(ZoneId zone);

    @Positive
    public static Clock tickSeconds(ZoneId zone);

    @Positive
    public static Clock tickMinutes(ZoneId zone);

    @Positive
    public static Clock tick(Clock baseClock, Duration tickDuration);

    @Positive
    public static Clock fixed(Instant fixedInstant, ZoneId zone);

    @Positive
    public static Clock offset(Clock baseClock, Duration offsetDuration);

    @Positive
    protected Clock() {
    @Positive
    }

    @Positive
    public abstract ZoneId getZone();

    @Positive
    @Override
    @Positive
    public abstract Clock withZone(ZoneId zone);

    @Positive
    @Override
    @Positive
    public long millis();

    @Positive
    @Override
    @Positive
    public abstract Instant instant();

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
    static Instant currentInstant();

    @Positive
    static final class SystemInstantSource implements InstantSource, Serializable {

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class SystemClock extends Clock implements Serializable {

    @Positive
        @Override
    @Positive
        public ZoneId getZone();

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class FixedClock extends Clock implements Serializable {

    @Positive
        @Override
    @Positive
        public ZoneId getZone();

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class OffsetClock extends Clock implements Serializable {

    @Positive
        @Override
    @Positive
        public ZoneId getZone();

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class TickClock extends Clock implements Serializable {

    @Positive
        @Override
    @Positive
        public ZoneId getZone();

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class SourceClock extends Clock implements Serializable {

    @Positive
        @Override
    @Positive
        public ZoneId getZone();

    @Positive
        @Override
    @Positive
        public Clock withZone(ZoneId zone);

    @Positive
        @Override
    @Positive
        public long millis();

    @Positive
        @Override
    @Positive
        public Instant instant();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
