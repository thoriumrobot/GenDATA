/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package jdk.internal.org.objectweb.asm.tree;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import jdk.internal.org.objectweb.asm.MethodVisitor;

    @Positive
public class InsnList implements Iterable<AbstractInsnNode> {

    @Positive
    public int size();

    @Positive
    public AbstractInsnNode getFirst();

    @Positive
    public AbstractInsnNode getLast();

    @Positive
    public AbstractInsnNode get(final int index);

    @Positive
    @Pure
    @Positive
    public boolean contains(final AbstractInsnNode insnNode);

    @Positive
    public int indexOf(final AbstractInsnNode insnNode);

    @Positive
    public void accept(final MethodVisitor methodVisitor);

    @Positive
    @Override
    @Positive
    public ListIterator<AbstractInsnNode> iterator();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public ListIterator<AbstractInsnNode> iterator(final int index);

    @Positive
    public AbstractInsnNode[] toArray();

    @Positive
    public void set(final AbstractInsnNode oldInsnNode, final AbstractInsnNode newInsnNode);

    @Positive
    public void add(final AbstractInsnNode insnNode);

    @Positive
    public void add(final InsnList insnList);

    @Positive
    public void insert(final AbstractInsnNode insnNode);

    @Positive
    public void insert(final InsnList insnList);

    @Positive
    public void insert(final AbstractInsnNode previousInsn, final AbstractInsnNode insnNode);

    @Positive
    public void insert(final AbstractInsnNode previousInsn, final InsnList insnList);

    @Positive
    public void insertBefore(final AbstractInsnNode nextInsn, final AbstractInsnNode insnNode);

    @Positive
    public void insertBefore(final AbstractInsnNode nextInsn, final InsnList insnList);

    @Positive
    public void remove(final AbstractInsnNode insnNode);

    @Positive
    void removeAll(final boolean mark);

    @Positive
    public void clear();

    @Positive
    public void resetLabels();

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    private final class InsnListIterator implements ListIterator {

    @Positive
        @Override
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        public Object next();

    @Positive
        @Override
    @Positive
        public void remove();

    @Positive
        @Override
    @Positive
        public boolean hasPrevious();

    @Positive
        @Override
    @Positive
        public Object previous();

    @Positive
        @Override
    @Positive
        public int nextIndex();

    @Positive
        @Override
    @Positive
        public int previousIndex();

    @Positive
        @Override
    @Positive
        public void add(final Object o);

    @Positive
        @Override
    @Positive
        public void set(final Object o);
    @Positive
    }
    @Positive
}
