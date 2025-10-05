/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class WaitDispatchSupport {
/*
    @Copyright * Positive (c) 2010, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Timer;
    @Positive
import java.util.TimerTask;
    @Positive
import java.util.concurrent.atomic.AtomicBoolean;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.AccessController;
    @Positive
import sun.awt.PeerEvent;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class WaitDispatchSupport implements SecondaryLoop {

    @Positive
    public WaitDispatchSupport(EventDispatchThread dispatchThread) {
    @Positive
    }

    @Positive
    public WaitDispatchSupport(EventDispatchThread dispatchThread, Conditional extCond) {
    @Positive
    }

    @Positive
    public WaitDispatchSupport(EventDispatchThread dispatchThread, Conditional extCondition, EventFilter filter, long interval) {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Override
    @Positive
    public boolean enter();

    @Positive
    public boolean exit();
    @Positive
}

}