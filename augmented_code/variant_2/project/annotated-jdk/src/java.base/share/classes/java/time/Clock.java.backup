/*
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectStreamException;
import static java.time.LocalTime.NANOS_PER_MINUTE;
import static java.time.LocalTime.NANOS_PER_SECOND;
import static java.time.LocalTime.NANOS_PER_MILLI;
import java.io.Serializable;
import java.util.Objects;
import java.util.TimeZone;
import jdk.internal.misc.VM;

public abstract class Clock implements InstantSource {

    public static Clock systemUTC();

    public static Clock systemDefaultZone();

    public static Clock system(ZoneId zone);

    public static Clock tickMillis(ZoneId zone);

    public static Clock tickSeconds(ZoneId zone);

    public static Clock tickMinutes(ZoneId zone);

    public static Clock tick(Clock baseClock, Duration tickDuration);

    public static Clock fixed(Instant fixedInstant, ZoneId zone);

    public static Clock offset(Clock baseClock, Duration offsetDuration);

    protected Clock() {
    }

    public abstract ZoneId getZone();

    @Override
    public abstract Clock withZone(ZoneId zone);

    @Override
    public long millis();

    @Override
    public abstract Instant instant();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    static Instant currentInstant();

    static final class SystemInstantSource implements InstantSource, Serializable {

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static final class SystemClock extends Clock implements Serializable {

        @Override
        public ZoneId getZone();

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static final class FixedClock extends Clock implements Serializable {

        @Override
        public ZoneId getZone();

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static final class OffsetClock extends Clock implements Serializable {

        @Override
        public ZoneId getZone();

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static final class TickClock extends Clock implements Serializable {

        @Override
        public ZoneId getZone();

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static final class SourceClock extends Clock implements Serializable {

        @Override
        public ZoneId getZone();

        @Override
        public Clock withZone(ZoneId zone);

        @Override
        public long millis();

        @Override
        public Instant instant();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }
}
